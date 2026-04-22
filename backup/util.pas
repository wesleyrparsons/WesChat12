unit Util;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Matrix;

procedure InitRoPE(var InvFreq: TFVector; ModelDim: Integer);
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
procedure InitializeTransformer(var WModel: TWModelParams);
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure UpdateParam(const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
procedure Optimization(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure ApplyRoPE(var H: TSeqMatrix;  const InvFreq: TFVector; SeqLen, ModelDim: Integer);
procedure ApplyAutoregressiveMask(var ScoresHead: TScoresMatrix; const L: Integer);
procedure SoftmaxForward(const x: TFVector; out y: array of Single);
procedure SoftmaxBackward(const y, dy:  TFVector; out dx: array of Single);
procedure LayerNormForward(const InX: TSeqMatrix; var OutX: TSeqMatrix; SeqLen: Integer;
  const Gamma, Beta: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
procedure LayerNormBackward(const dY: TSeqMatrix; var dX: TSeqMatrix; var dGamma, dBeta: TSeqVector;
  SeqLen: Integer; const Gamma: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
procedure GradientFromKLDivergence(var WModelState: TWModelState);
procedure GradientFromCEProbabilities(var WModelParams: TWModelParams);
procedure BackpropAdd(const dOut: TSeqMatrix; var dA, dB: TSeqMatrix; const L, D: Integer);

implementation

// Initialize test vector.
procedure InitTestVector(var N: TFSVector);           // Test procedure, not used.
var
  i: Integer;
begin
for i := 0 to SeqLen - 1 do
  N[i] := 0.0;
end;

// Apply at initialization of transformer.
procedure InitRoPE(var InvFreq: TFVector; ModelDim: Integer);
var
  j: Integer;
begin
  // ModelDim must be even.
  SetLength(InvFreq, ModelDim div 2);
  for j := 0 to (ModelDim div 2) - 1 do
    InvFreq[j] := Exp( - (2.0 * j) / ModelDim * Ln(10000.0) );
end;

// Xavier-Glorot initialization on W0 matrix.
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on WHead matrix.
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to HeadDim - 1 do
    for j := 0 to HeadDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W1 matrix.
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W2 matrix.
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[j, i] := (2 * r - 1) * Limit;
    end;
end;

// Initialize the transformer stage.
procedure InitializeTransformer(var WModel: TWModelParams);
var
  j: Integer;
begin
  with WModel do begin
    // InitTestVector(TestVector);
    // Initialize RoPE.
    InitRoPE(InvFreq, ModelDim);

    // Initialize weight matrix W0.
    XGUniformW(W0.Value, ModelDim, ModelDim);

    // Initialize the weights with Xavier-Glorot function.
    XGUniformW(Wq.Value, ModelDim, ModelDim);
    XGUniformW(Wk.Value, ModelDim, ModelDim);
    XGUniformW(Wv.Value, ModelDim, ModelDim);

    // Initialize W1 and W2 weight matrices.
    XGUniformW1(W1.Value, ModelDim, ModelDimProj);
    XGUniformW2(W2.Value, ModelDimProj, ModelDim);

    // Initialize WVocab weight matrices.
    //XGUniformWVocab(Embeddings.Value, nVocab, ModelDim);

    // Initialize b1 and b2.
    FillChar(b1.Value, SizeOf(b1.Value), 0);
    FillChar(b2.Value, SizeOf(b2.Value), 0);

    // Initialize Beta and Gamma, LN 1 and 2, with SD and mean.
    FillChar(Beta1.Value, SizeOf(Beta1.Value), 0);
    FillChar(Beta2.Value, SizeOf(Beta2.Value), 0);
    for j := 0 to ModelDim - 1 do begin
      Gamma1.Value[j] := 1.0;
      Gamma2.Value[j] := 1.0;
    end;
  end;
end;

// Zero out all gradients.
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  with WModelState do begin
    FillChar(X.Grad, SizeOf(X.Grad), 0);
    FillChar(X1.Grad, SizeOf(X1.Grad), 0);
    FillChar(X2.Grad, SizeOf(X2.Grad), 0);
    FillChar(X3.Grad, SizeOf(X3.Grad), 0);
    FillChar(X4.Grad, SizeOf(X4.Grad), 0);
    FillChar(X5.Grad, SizeOf(X5.Grad), 0);
    FillChar(X6.Grad, SizeOf(X6.Grad), 0);
    FillChar(X7.Grad, SizeOf(X7.Grad), 0);
    FillChar(X1k.Grad, SizeOf(X1k.Grad), 0);
    FillChar(X1q.Grad, SizeOf(X1q.Grad), 0);
    FillChar(X1v.Grad, SizeOf(X1v.Grad), 0);
    FillChar(K.Grad, SizeOf(K.Grad), 0);
    FillChar(Q.Grad, SizeOf(Q.Grad), 0);
    FillChar(V.Grad, SizeOf(V.Grad), 0);
    FillChar(Hidden1.Grad, SizeOf(Hidden1.Grad), 0);
    FillChar(Hidden2.Grad, SizeOf(Hidden2.Grad), 0);
    with WModelParams do begin
      FillChar(Wk.Grad, SizeOf(Wk.Grad), 0);
      FillChar(Wq.Grad, SizeOf(Wq.Grad), 0);
      FillChar(Wv.Grad, SizeOf(Wv.Grad), 0);
      FillChar(W0.Grad, SizeOf(W0.Grad), 0);
      FillChar(W1.Grad, SizeOf(W1.Grad), 0);
      FillChar(W2.Grad, SizeOf(W2.Grad), 0);
      FillChar(Embeddings.Grad, SizeOf(Embeddings.Grad), 0);
      FillChar(b1.Grad, SizeOf(b1.Grad), 0);
      FillChar(b2.Grad, SizeOf(b2.Grad), 0);
      FillChar(Gamma1.Grad, SizeOf(Gamma1.Grad), 0);
      FillChar(Gamma2.Grad, SizeOf(Gamma2.Grad), 0);
      FillChar(Beta1.Grad, SizeOf(Beta1.Grad), 0);
      FillChar(Beta2.Grad, SizeOf(Beta2.Grad), 0);
    end;
  end;
end;

// Parameter update. Param := Param - LearningRate * Grad.
procedure UpdateParam(const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
begin
  AddScaled(N, -LearningRate, Grad, Param);
end;

{ Optimization }
// Update the weights and biases.
procedure Optimization(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  i, v: Integer;
begin
  with WModelParams do begin
    // W0 weights: main attention output.
    UpdateParam(ModelDim * ModelDim, LearningRate, @W0.Grad[0,0], @W0.Value[0,0]);
    //cblas_saxpy(ModelDim * ModelDim, -LearningRate, @W0.Grad[0, 0], 1, @W0.Value[0, 0], 1);

    // Wq, Wk, Wv weights: Q, K, V.
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wq.Grad[0,0], @Wq.Value[0,0]);
    //cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wq.Grad[0, 0], 1, @Wq.Value[0, 0], 1);
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wk.Grad[0,0], @Wk.Value[0,0]);
    //cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wk.Grad[0, 0], 1, @Wk.Value[0, 0], 1);
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wv.Grad[0,0], @Wv.Value[0,0]);
    //cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wv.Grad[0, 0], 1, @Wv.Value[0, 0], 1);

    // W1, W2: feed-forward and vocab projection.
    UpdateParam(ModelDim * ModelDim, LearningRate, @W1.Grad[0,0], @W1.Value[0,0]);
    //cblas_saxpy(ModelDim * ModelDimProj, -LearningRate, @W1.Grad[0, 0], 1, @W1.Value[0, 0], 1);
    UpdateParam(ModelDim * ModelDim, LearningRate, @W2.Grad[0,0], @W2.Value[0,0]);
    //cblas_saxpy(ModelDimProj * ModelDim, -LearningRate, @W2.Grad[0, 0], 1, @W2.Value[0, 0], 1);

    // b1, b2: biases.
    UpdateParam(ModelDim, LearningRate, @b1.Grad[0], @b1.Value[0]);
    //cblas_saxpy(ModelDimProj, -LearningRate, @b1.Grad[0], 1, @b1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @b2.Grad[0], @b2.Value[0]);
    //cblas_saxpy(ModelDim, -LearningRate, @b2.Grad[0], 1, @b2.Value[0], 1);

    // Gamma1, Gamm2, Beta1, Beta2: Layer-Norm parameters.
    UpdateParam(ModelDim, LearningRate, @Gamma1.Grad[0], @Gamma1.Value[0]);
    //cblas_saxpy(ModelDim, -LearningRate, @Gamma1.Grad[0], 1, @Gamma1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Gamma2.Grad[0], @Gamma2.Value[0]);
    //cblas_saxpy(ModelDim, -LearningRate, @Gamma2.Grad[0], 1, @Gamma2.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Beta1.Grad[0], @Beta1.Value[0]);
    //cblas_saxpy(ModelDim, -LearningRate, @Beta1.Grad[0], 1, @Beta1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Beta2.Grad[0], @Beta2.Value[0]);
    //cblas_saxpy(ModelDim, -LearningRate, @Beta2.Grad[0], 1, @Beta2.Value[0], 1);

    // Embeddings.
    // Add input-side embedding gradients into Embeddings.Grad.
    for i := 0 to SeqLen - 1 do begin
      v := TokenID[i];
      AddScaled(ModelDim, 1.0, @WModelState.X.Grad[i,0], @Embeddings.Grad[v,0]);
      //cblas_saxpy(ModelDim, 1.0, @WModelState.X.Grad[i,0], 1, @Embeddings.Grad[v,0], 1);
    end;

    // Apply the total embedding gradient (output-side + input-side).
    UpdateParam(nVocab * ModelDim, LearningRate, @Embeddings.Grad[0,0], @Embeddings.Value[0,0]);
    //cblas_saxpy(nVocab * ModelDim, -LearningRate, @Embeddings.Grad[0,0], 1, @Embeddings.Value[0,0], 1);
    {for i := 0 to SeqLen - 1 do begin
      v := TokenID[i];    // Same as TokenizedCorpus;
      cblas_saxpy(ModelDim, -LearningRate, @X.Grad[i,0], 1, @Embeddings.Value[v, 0], 1);
    end;}
  end;
end;

// Rotary positional encoding.
// Apply RoPE to both Q and K, [0..SeqLen - 1, 0..ModelDim - 1]
// Apply before head-splitting, immediately after computing Q and K.
procedure ApplyRoPE(var H: TSeqMatrix;  const InvFreq: TFVector; SeqLen, ModelDim: Integer);
var
  i, j: Integer;
  Angle, c, s, x0, x1: Single;
begin
  for i := 0 to SeqLen - 1 do
    for j := 0 to (ModelDim div 2) - 1 do begin
      Angle := i * InvFreq[j];
      c := Cos(Angle);
      s := Sin(Angle);

      // Original pair.
      x0 := H[i, 2 * j];
      x1 := H[i, 2 * j + 1];

      // Rotated pair.
      H[i, 2 * j]   :=  x0 * c - x1 * s;
      H[i, 2 * j + 1] :=  x0 * s + x1 * c;
    end;
end;

// Simple autoregressive masking.
procedure ApplyAutoregressiveMask(var ScoresHead: TScoresMatrix; const L: Integer);
var
  i, j: Integer;
const
  NEG_INF: Single = -1e30;
begin
  for i := 0 to L - 1 do
    for j := i + 1 to L - 1 do
      ScoresHead[i, j] := NEG_INF;
end;

// Softmax procedure forward.
procedure SoftmaxForward(const x: TFVector; out y: array of Single);
var
  i: Integer;
  MaxVal, SumVal, InvT: Single;
begin
  // Find max for numerical stability.
  InvT := 1.0 / Temperature;
  MaxVal := x[0] * InvT;
  for i := 1 to High(x) do
    if (x[i] * InvT) > MaxVal then
      MaxVal := x[i] * InvT;

  // Compute exp(x - max).
  SumVal := 0;
  for i := 0 to High(x) do begin
    y[i] := Exp((x[i] * InvT) - MaxVal);
    SumVal := SumVal + y[i];
  end;

  // Normalize.
  SumVal := 1.0 / SumVal;
  for i := 0 to High(x) do
    y[i] := y[i] * SumVal;
end;

// Softmax procedure backward.
procedure SoftmaxBackward(const y, dy:  TFVector; out dx: array of Single);
var
  j, D: Integer;
  dot: Single;
begin
  D := Length(y);

  // dot = sum_j dy[j] * y[j].
  dot := 0.0;
  for j := 0 to D - 1 do
    dot := dot + dy[j] * y[j];

  // dx[j] = y[j] * (dy[j] - dot).
  for j := 0 to D - 1 do
    dx[j] := y[j] * (dy[j] - dot);
end;

// Layer-Norm matrix.
procedure LayerNormForward(const InX: TSeqMatrix; var OutX: TSeqMatrix; SeqLen: Integer;
  const Gamma, Beta: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  MeanL, VarL, InvStd: Single;
const
  EPS = 1e-5;
begin
  for i := 0 to SeqLen - 1 do begin
    MeanL := 0.0;
    for j := 0 to ModelDim - 1 do
      MeanL := MeanL + InX[i, j];
    MeanL := MeanL / ModelDim;

    VarL := 0.0;
    for j := 0 to ModelDim - 1 do
      VarL := VarL + Sqr(InX[i, j] - MeanL);
    VarL := VarL / ModelDim;

    InvStd := 1.0 / Sqrt(VarL + EPS);

    for j := 0 to ModelDim - 1 do
      OutX[i, j] := (InX[i, j] - MeanL) * InvStd * Gamma[j] + Beta[j];
    LNInvStd[i] := InvStd;
    for j := 0 to ModelDim - 1 do
      LNXhat[i, j] := (InX[i, j] - MeanL) * InvStd;
  end;
end;

// Layer-Norm matrix on back propagation. dY is upstream gradient. dX is output gradient.
// dGamma, dBeta are accumulated over all rows.
procedure LayerNormBackward(const dY: TSeqMatrix; var dX: TSeqMatrix; var dGamma, dBeta: TSeqVector;
  SeqLen: Integer; const Gamma: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  sum1, sum2, scale: Single;
  dHat: TSeqVector;
begin
  for i := 0 to SeqLen - 1 do begin
    // Step 1: dHat = dY * Gamma.
    sum1 := 0.0;
    sum2 := 0.0;
    for j := 0 to ModelDim - 1 do begin
      dHat[j] := dY[i, j] * Gamma[j];
      sum1 := sum1 + dHat[j];
      sum2 := sum2 + dHat[j] * LNXhat[i][j];
    end;

    // Step 2: compute dX.
    scale := LNInvStd[i] / ModelDim;
    for j := 0 to ModelDim - 1 do
      dX[i, j] := scale * (ModelDim * dHat[j] - sum1 - LNXhat[i, j] * sum2);

    // Step 3: accumulate dGamma and dBeta.
    for j := 0 to ModelDim - 1 do begin
      dGamma[j] := dGamma[j] + dY[i, j] * LNXhat[i][j];
      dBeta[j]  := dBeta[j]  + dY[i, j];
    end;
  end;
end;

// Calculate cross-entropy gradient from probabilities and target, one-hot.
procedure GradientFromCEProbabilities(var WModelState: TWModelState);
var
  i, s: Integer;
begin
  with WModelState do
    for i := 0 to SeqLen - 1 do begin
      for s := 0 to nVocab - 1 do
        TopGradient[i, s] := Probs[i, s];

      TopGradient[i, TargetTokens[i]] :=
        Probs[i, TargetTokens[i]] - 1.0;
    end;
end;


// Calculate gradient for KL divergence with one-hot targets: dL/dProbs = Q - P.
procedure GradientFromKLDivergence(var WModelState: TWModelState);
var
  i, s: Integer;
begin
  with WModelState do
    for i := 0 to SeqLen - 1 do
      for s := 0 to nVocab - 1 do
        if s = TargetTokens[i] then
          TopGradient[i, s] := Probs[i, s] - 1.0   // Q - 1.
        else
          TopGradient[i, s] := Probs[i, s] - 0.0;  // Q - 0.
end;

// Back propagation addition.
procedure BackpropAdd(const dOut: TSeqMatrix; var dA, dB: TSeqMatrix; const L, D: Integer);
var
  i, j: Integer;
begin
  for i := 0 to L - 1 do
    for j := 0 to D - 1 do begin
      dA[i, j] := dA[i, j] + dOut[i, j];
      dB[i, j] := dB[i, j] + dOut[i, j];
    end;
end;

end.

