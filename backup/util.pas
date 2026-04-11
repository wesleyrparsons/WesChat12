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
procedure XGUniformWVocab(var W: TVocabWeightMatrix; FanIn, FanOut: Integer);
procedure InitializeTransformer(var WesModel: ModelType);
procedure ZeroGradients(var WesModel: ModelType);
procedure Optimization(var WesModel: ModelType);

implementation

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

// Xavier-Glorot initialization on WVocab matrix.
procedure XGUniformWVocab(var W: TVocabWeightMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to nVocab - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Initialize the transformer stage.
procedure InitializeTransformer(var WesModel: ModelType);
var
  j: Integer;
begin
  with WesModel do begin
    // InitTestVector(TestVector);
    // Initialize RoPE.
    InitRoPE(InvFreq, HeadDim);

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
    XGUniformWVocab(WVocab.Value, ModelDim, nVocab);

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
procedure ZeroGradients(var WesModel: ModelType);
begin
  FillChar(X.Grad, SizeOf(X.Grad), 0);
  FillChar(X1.Grad, SizeOf(X1.Grad), 0);
  FillChar(X2.Grad, SizeOf(X2.Grad), 0);
  FillChar(X3.Grad, SizeOf(X3.Grad), 0);
  FillChar(X4.Grad, SizeOf(X4.Grad), 0);
  FillChar(X5.Grad, SizeOf(X5.Grad), 0);
  FillChar(X6.Grad, SizeOf(X6.Grad), 0);
  FillChar(X7.Grad, SizeOf(X7.Grad), 0);
  FillChar(X8.Grad, SizeOf(X8.Grad), 0);
  with WesModel do begin
    FillChar(W0.Grad, SizeOf(W0.Grad), 0);
    FillChar(W1.Grad, SizeOf(W1.Grad), 0);
    FillChar(W2.Grad, SizeOf(W2.Grad), 0);
    FillChar(WVocab.Grad, SizeOf(WVocab.Grad), 0);
    FillChar(b1.Grad, SizeOf(b1.Grad), 0);
    FillChar(b2.Grad, SizeOf(b2.Grad), 0);
    FillChar(Gamma1.Grad, SizeOf(Gamma1.Grad), 0);
    FillChar(Gamma2.Grad, SizeOf(Gamma2.Grad), 0);
    FillChar(Beta1.Grad, SizeOf(Beta1.Grad), 0);
    FillChar(Beta2.Grad, SizeOf(Beta2.Grad), 0);
  end;
end;

{ Optimization }
// Update the weights and biases.
procedure Optimization(var WesModel: ModelType);
var
  i, v: Integer;
begin
  with WesModel do begin
    // W0 weights: main attention output.
    cblas_saxpy(ModelDim * ModelDim, -LearningRate, @W0.Grad[0, 0], 1, @W0.Value[0, 0], 1);

    // Wq, Wk, Wv weights: Q, K, V.
    cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wq.Grad[0, 0], 1, @Wq.Value[0, 0], 1);
    cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wk.Grad[0, 0], 1, @Wk.Value[0, 0], 1);
    cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wv.Grad[0, 0], 1, @Wv.Value[0, 0], 1);

    // W1, W2: feed-forward and vocab projection.
    cblas_saxpy(ModelDim * ModelDimProj, -LearningRate, @W1.Grad[0, 0], 1, @W1.Value[0, 0], 1);
    cblas_saxpy(ModelDimProj * ModelDim, -LearningRate, @W2.Grad[0, 0], 1, @W2.Value[0, 0], 1);
    cblas_saxpy(ModelDim * nVocab, -LearningRate, @WVocab.Grad[0, 0], 1, @WVocab.Value[0, 0], 1);

    // b1, b2: biases.
    cblas_saxpy(ModelDimProj, -LearningRate, @b1.Grad[0], 1, @b1.Value[0], 1);
    cblas_saxpy(ModelDim, -LearningRate, @b2.Grad[0], 1, @b2.Value[0], 1);

    // Gamma1, Gamm2, Beta1, Beta2: Layer-Norm parameters.
    cblas_saxpy(ModelDim, -LearningRate, @Gamma1.Grad[0], 1, @Gamma1.Value[0], 1);
    cblas_saxpy(ModelDim, -LearningRate, @Gamma2.Grad[0], 1, @Gamma2.Value[0], 1);
    cblas_saxpy(ModelDim, -LearningRate, @Beta1.Grad[0], 1, @Beta1.Value[0], 1);
    cblas_saxpy(ModelDim, -LearningRate, @Beta2.Grad[0], 1, @Beta2.Value[0], 1);

    // Embeddings.
    for i := 0 to SeqLen - 1 do begin
      v := TokenID[i];    // Same as TokenizedCorpus;
      cblas_saxpy(ModelDim, -LearningRate, @X.Grad[i,0], 1, @Embeddings[0, v], 1);
    end;
  end;
end;

end.

