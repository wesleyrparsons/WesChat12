unit Util;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Matrix;

const
  WeightSize: Integer = ModelDim * ModelDim * SizeOf(Single);
  WeightProjSize: Integer = ModelDim * ModelDimProj * SizeOf(Single);
  bProjSize: Integer = ModelDimProj * SizeOf(Single);
  bSize: Integer = ModelDim * SizeOf(Single);
  SeqSize: Integer = SeqLen * SizeOf(Single);
  ModelSize: Integer = ModelDim * SizeOf(Single);
  XSize: Integer = SeqLen * ModelDim * SizeOf(Single);
  HiddenSize: Integer = SeqLen * ModelDimProj * SizeOf(Single);
  ScoresSize: Integer = SeqLen * SeqLen * SizeOf(Single);
  EmbeddingsSize: Integer = DimVocab * ModelDim * SizeOf(Single);
  InvFreqSize: Integer = (ModelDim div 2) * SizeOf(Single);
  ProbsSize: Integer = SeqLen * DimVocab * SizeOf(Single);

procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
procedure InitializeTransformer(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
procedure UpdateParam(const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
procedure Optimization(var WModelParams: TWModelParams; const Blk: Integer);
procedure LaunchEmbeddingLookup(Embeddings: PSingle; InputTokens: PInteger; X: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure UpdateEmbeddings(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens: TIDimVector);
procedure ApplyRoPE(var H: TSeqMatrix;  const InvFreq: TFVector; SeqLen, ModelDim: Integer);
procedure ApplyAutoRegressiveMask(var ScoresHead: TScoresMatrix; const L: Integer);
procedure LaunchAutoRegressiveMask(Scores: PSingle; SeqLen: Integer); cdecl; external 'WesChatKernel12.dll';
procedure LaunchDropout(X: PSingle; N: Integer; DropProb: Single; Seed: UInt64); cdecl; external 'WesChatKernel12.dll';
procedure SoftmaxForwardN(const x: PSingle; y: PSingle; const N: Integer);
procedure LaunchSoftmaxForwardN(X: PSingle; Y: PSingle; Rows: Integer; N: Integer; Temperature: Single);
  cdecl; external 'WesChatKernel12.dll';
procedure SoftmaxBackward(const y, dy:  TFVector; out dx: array of Single);
procedure LayerNormForward(const InX: TSeqMatrix; var OutX: TSeqMatrix; SeqLen: Integer;
  const Gamma, Beta: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
procedure LaunchLayerNormForward(InX, OutX, Gamma, Beta, LNXhat, LNInvStd: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LayerNormBackward(const dY: TSeqMatrix; var dX: TSeqMatrix; var dGamma, dBeta: TSeqVector;
  SeqLen: Integer; const Gamma: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
procedure LaunchRoPEForward(H: PSingle; InvFreq: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure GradientFromCEProbabilities(var WModelState: TWModelState);
procedure LaunchCEGradient(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; SeqLen: Integer; nVocab: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure GradientFromKLDivergence(var WModelState: TWModelState);
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

// Allocate cublas memory.       Separate for State and Params?
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  h, k: Integer;
begin
  // Input and target tokens.
  cudaMalloc(@dInputTokens, SeqLen * SizeOf(Integer));
  cudaMalloc(@dTargetTokens, SeqLen * SizeOf(Integer));

  // Global/shared parameters.
  with WModelParams do begin
    cudaMalloc(@Embeddings.dValue, EmbeddingsSize);
    cudaMalloc(@Embeddings.dGrad, EmbeddingsSize);
  end;
  with WModelState do begin
    cudaMalloc(@dInvFreq, InvFreqSize);
    cudaMalloc(@dProbs, ProbsSize);
    cudaMalloc(@dTopGradient, ProbsSize);
  end;

  // Per block parameters.
  for k := 0 to nBlock - 1 do begin
    with WModelParams.ParamBlock[k] do begin
      cudaMalloc(@Wq.dValue, WeightSize);
      cudaMalloc(@Wq.dGrad, WeightSize);
      cudaMalloc(@Wk.dValue, WeightSize);
      cudaMalloc(@Wk.dGrad, WeightSize);
      cudaMalloc(@Wv.dValue, WeightSize);
      cudaMalloc(@Wv.dGrad, WeightSize);
      cudaMalloc(@W0.dValue, WeightSize);
      cudaMalloc(@W0.dGrad, WeightSize);
      cudaMalloc(@W1.dValue, WeightProjSize);
      cudaMalloc(@W1.dGrad, WeightProjSize);
      cudaMalloc(@W2.dValue, WeightProjSize);
      cudaMalloc(@W2.dGrad, WeightProjSize);
      cudaMalloc(@b1.dValue, bProjSize);
      cudaMalloc(@b1.dGrad, bProjSize);
      cudaMalloc(@b2.dValue, bSize);
      cudaMalloc(@b2.dGrad, bSize);
      cudaMalloc(@Gamma1.dValue, ModelSize);
      cudaMalloc(@Gamma1.dGrad, ModelSize);
      cudaMalloc(@Beta1.dValue, ModelSize);
      cudaMalloc(@Beta1.dGrad, ModelSize);
      cudaMalloc(@Gamma2.dValue, ModelSize);
      cudaMalloc(@Gamma2.dGrad, ModelSize);
      cudaMalloc(@Beta2.dValue, ModelSize);
      cudaMalloc(@Beta2.dGrad, ModelSize);
    end;
    with WModelState.StateBlock[k] do begin
      cudaMalloc(@X.dValue, XSize);
      cudaMalloc(@X.dGrad, XSize);
      cudaMalloc(@X1.dValue, XSize);
      cudaMalloc(@X1.dGrad, XSize);
      cudaMalloc(@X2.dValue, XSize);
      cudaMalloc(@X2.dGrad, XSize);
      cudaMalloc(@X3.dValue, XSize);
      cudaMalloc(@X3.dGrad, XSize);
      cudaMalloc(@X4.dValue, XSize);
      cudaMalloc(@X4.dGrad, XSize);
      cudaMalloc(@X5.dValue, XSize);
      cudaMalloc(@X5.dGrad, XSize);
      cudaMalloc(@X6.dValue, XSize);
      cudaMalloc(@X6.dGrad, XSize);
      cudaMalloc(@X7.dValue, XSize);
      cudaMalloc(@X7.dGrad, XSize);
      cudaMalloc(@X1q.dValue, XSize);
      cudaMalloc(@X1q.dGrad, XSize);
      cudaMalloc(@X1k.dValue, XSize);
      cudaMalloc(@X1k.dGrad, XSize);
      cudaMalloc(@X1v.dValue, XSize);
      cudaMalloc(@X1v.dGrad, XSize);
      cudaMalloc(@Q.dValue, XSize);
      cudaMalloc(@Q.dGrad, XSize);
      cudaMalloc(@K.dValue, XSize);
      cudaMalloc(@K.dGrad, XSize);
      cudaMalloc(@V.dValue, XSize);
      cudaMalloc(@V.dGrad, XSize);
      for h := 0 to nHead - 1 do begin
        cudaMalloc(@ScoresHead1[h].dValue, ScoresSize);
        cudaMalloc(@ScoresHead1[h].dGrad, ScoresSize);
        cudaMalloc(@ScoresHead2[h].dValue, ScoresSize);
        cudaMalloc(@ScoresHead2[h].dGrad, ScoresSize);
      end;
      cudaMalloc(@Hidden1.dValue, HiddenSize);
      cudaMalloc(@Hidden1.dGrad, HiddenSize);
      cudaMalloc(@Hidden2.dValue, HiddenSize);
      cudaMalloc(@Hidden2.dGrad, HiddenSize);
      cudaMalloc(@dLNInvStd1, SeqSize);
      cudaMalloc(@dLNXHat1, XSize);
      cudaMalloc(@dLNInvStd2, SeqSize);
      cudaMalloc(@dLNXHat2, XSize);
    end;
  end;
end;

// De-allocate cublas memory.
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  h, k: Integer;
begin
  cudaFree(@dInputTokens);
  cudaFree(@dTargetTokens);

  with WModelParams do begin
    cudaFree(@Embeddings.dValue);
    cudaFree(@Embeddings.dGrad);
  end;
  with WModelState do begin
    cudaFree(@dInvFreq);
    cudaFree(@Probs);
    cudaFree(@TopGradient);
  end;

  for k := 0 to nBlock - 1 do begin
    with WModelParams.ParamBlock[k] do begin
      cudaFree(@Wq.dValue);
      cudaFree(@Wq.dGrad);
      cudaFree(@Wk.dValue);
      cudaFree(@Wk.dGrad);
      cudaFree(@Wv.dValue);
      cudaFree(@Wv.dGrad);
      cudaFree(@W0.dValue);
      cudaFree(@W0.dGrad);
      cudaFree(@W1.dValue);
      cudaFree(@W1.dGrad);
      cudaFree(@W2.dValue);
      cudaFree(@W2.dGrad);
      cudaFree(@b1.dValue);
      cudaFree(@b1.dGrad);
      cudaFree(@b2.dValue);
      cudaFree(@b2.dGrad);
      cudaFree(@Gamma1.dValue);
      cudaFree(@Gamma1.dGrad);
      cudaFree(@Beta1.dValue);
      cudaFree(@Beta1.dGrad);
      cudaFree(@Gamma2.dValue);
      cudaFree(@Gamma2.dGrad);
      cudaFree(@Beta2.dValue);
      cudaFree(@Beta2.dGrad);
    end;
    with WModelState.StateBlock[k] do begin
      cudaFree(@X.dValue);
      cudaFree(@X.dGrad);
      cudaFree(@X1.dValue);
      cudaFree(@X1.dGrad);
      cudaFree(@X2.dValue);
      cudaFree(@X2.dGrad);
      cudaFree(@X3.dValue);
      cudaFree(@X3.dGrad);
      cudaFree(@X4.dValue);
      cudaFree(@X4.dGrad);
      cudaFree(@X5.dValue);
      cudaFree(@X5.dGrad);
      cudaFree(@X6.dValue);
      cudaFree(@X6.dGrad);
      cudaFree(@X7.dValue);
      cudaFree(@X7.dGrad);
      cudaFree(@X1q.dValue);
      cudaFree(@X1q.dGrad);
      cudaFree(@X1k.dValue);
      cudaFree(@X1k.dGrad);
      cudaFree(@X1v.dValue);
      cudaFree(@X1v.dGrad);
      cudaFree(@Q.dValue);
      cudaFree(@Q.dGrad);
      cudaFree(@K.dValue);
      cudaFree(@K.dGrad);
      cudaFree(@V.dValue);
      cudaFree(@V.dGrad);
      for h := 0 to nHead - 1 do begin
        cudaFree(@ScoresHead1[h].dValue);
        cudaFree(@ScoresHead1[h].dGrad);
        cudaFree(@ScoresHead2[h].dValue);
        cudaFree(@ScoresHead2[h].dGrad);
      end;
      cudaFree(@Hidden1.dValue);
      cudaFree(@Hidden1.dGrad);
      cudaFree(@Hidden2.dValue);
      cudaFree(@Hidden2.dGrad);
      cudaFree(@dLNInvStd1);
      cudaFree(@dLNXHat1);
      cudaFree(@dLNInvStd2);
      cudaFree(@dLNXHat2);
    end;
  end;
end;

// Copy parameters from host to device.
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
var
  k: Integer;
begin
  // Embeddings (global).
  cudaMemcpy(WModelParams.Embeddings.dValue, @WModelParams.Embeddings.Value[0,0], EmbeddingsSize, cudaMemcpyHostToDevice);

  // Other.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin

      // Attention weights.
      cudaMemcpy(Wq.dValue, @Wq.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wk.dValue, @Wk.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wv.dValue, @Wv.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W0.dValue, @W0.Value[0,0], WeightSize, cudaMemcpyHostToDevice);

      // MLP.
      cudaMemcpy(W1.dValue, @W1.Value[0,0], WeightProjSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dValue, @W2.Value[0,0], WeightProjSize, cudaMemcpyHostToDevice);

      // Biases.
      cudaMemcpy(b1.dValue, @b1.Value[0], bProjSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b2.dValue, @b2.Value[0], bSize, cudaMemcpyHostToDevice);

      // LayerNorm.
      cudaMemcpy(Gamma1.dValue, @Gamma1.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta1.dValue,  @Beta1.Value[0],  ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma2.dValue, @Gamma2.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta2.dValue,  @Beta2.Value[0],  ModelSize, cudaMemcpyHostToDevice);
  end;
end;

// Copy InvFreq from host to device.
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
begin
  cudaMemcpy(WModelState.dInvFreq, @WModelState.InvFreq[0], InvFreqSize, cudaMemcpyHostToDevice);
end;

// Initialize the transformer state stage.
procedure InitializeTransformer(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  j, k: Integer;
begin
  // May be able to delete second parameter above.
  // Do not zero the param grads -- thet is done in zero gradient.
  // Do not zero the state grads -- thet is done in zero gradient.
  // As to the param values -- they are zeroed below.
  // Do not zero the state values -- thet is not necessary.
  // Do I need to zero topgradient and prob.

  // Compute InvFreq.
  SetLength(WModelState.InvFreq, ModelDim div 2);
  for j := 0 to (ModelDim div 2) - 1 do     // ModelDim must be even.
    WModelState.InvFreq[j] := Exp( - (2.0 * j) / ModelDim * Ln(10000.0) );

  // Initialize param values.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin

      // Initialize weight matrix W0.
      XGUniformW(W0.Value, ModelDim, ModelDim);

      // Initialize the weights with Xavier-Glorot function.
      XGUniformW(Wq.Value, ModelDim, ModelDim);
      XGUniformW(Wk.Value, ModelDim, ModelDim);
      XGUniformW(Wv.Value, ModelDim, ModelDim);

      // Initialize W1 and W2 weight matrices.
      XGUniformW1(W1.Value, ModelDim, ModelDimProj);
      XGUniformW2(W2.Value, ModelDimProj, ModelDim);

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
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
begin
  with WModelState.StateBlock[Blk] do begin
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
  end;
  with WModelParams.ParamBlock[Blk] do begin
    FillChar(Wk.Grad, SizeOf(Wk.Grad), 0);
    FillChar(Wq.Grad, SizeOf(Wq.Grad), 0);
    FillChar(Wv.Grad, SizeOf(Wv.Grad), 0);
    FillChar(W0.Grad, SizeOf(W0.Grad), 0);
    FillChar(W1.Grad, SizeOf(W1.Grad), 0);
    FillChar(W2.Grad, SizeOf(W2.Grad), 0);
    FillChar(b1.Grad, SizeOf(b1.Grad), 0);
    FillChar(b2.Grad, SizeOf(b2.Grad), 0);
    FillChar(Gamma1.Grad, SizeOf(Gamma1.Grad), 0);
    FillChar(Gamma2.Grad, SizeOf(Gamma2.Grad), 0);
    FillChar(Beta1.Grad, SizeOf(Beta1.Grad), 0);
    FillChar(Beta2.Grad, SizeOf(Beta2.Grad), 0);
  end;
end;

// Parameter update. Param := Param - LearningRate * Grad.
procedure UpdateParam(const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
begin
  AddScaled(N, -LearningRate, Grad, Param); // Not efficient to use cublas here.
end;

{ Optimization }
// Update the weights and biases.
procedure Optimization(var WModelParams: TWModelParams; const Blk: Integer);
begin
  with WModelParams.ParamBlock[Blk] do begin
    // W0 weights: main attention output.
    UpdateParam(ModelDim * ModelDim, LearningRate, @W0.Grad[0,0], @W0.Value[0,0]);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @W0.Grad[0, 0], 1, @W0.Value[0, 0], 1);

    // Wq, Wk, Wv weights: Q, K, V.
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wq.Grad[0,0], @Wq.Value[0,0]);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wq.Grad[0, 0], 1, @Wq.Value[0, 0], 1);
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wk.Grad[0,0], @Wk.Value[0,0]);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wk.Grad[0, 0], 1, @Wk.Value[0, 0], 1);
    UpdateParam(ModelDim * ModelDim, LearningRate, @Wv.Grad[0,0], @Wv.Value[0,0]);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wv.Grad[0, 0], 1, @Wv.Value[0, 0], 1);

    // W1, W2: feed-forward and vocab projection.
    UpdateParam(ModelDim * ModelDimProj, LearningRate, @W1.Grad[0,0], @W1.Value[0,0]);
    UpdateParam(ModelDimProj * ModelDim, LearningRate, @W2.Grad[0,0], @W2.Value[0,0]);
    // cblas_saxpy(ModelDimProj * ModelDim, -LearningRate, @W2.Grad[0, 0], 1, @W2.Value[0, 0], 1);

    // b1, b2: biases.
    UpdateParam(ModelDimProj, LearningRate, @b1.Grad[0], @b1.Value[0]);
    // cblas_saxpy(ModelDimProj, -LearningRate, @b1.Grad[0], 1, @b1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @b2.Grad[0], @b2.Value[0]);
    // cblas_saxpy(ModelDim, -LearningRate, @b2.Grad[0], 1, @b2.Value[0], 1);

    // Gamma1, Gamm2, Beta1, Beta2: Layer-Norm parameters.
    UpdateParam(ModelDim, LearningRate, @Gamma1.Grad[0], @Gamma1.Value[0]);
    // cblas_saxpy(ModelDim, -LearningRate, @Gamma1.Grad[0], 1, @Gamma1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Gamma2.Grad[0], @Gamma2.Value[0]);
    // cblas_saxpy(ModelDim, -LearningRate, @Gamma2.Grad[0], 1, @Gamma2.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Beta1.Grad[0], @Beta1.Value[0]);
    // cblas_saxpy(ModelDim, -LearningRate, @Beta1.Grad[0], 1, @Beta1.Value[0], 1);
    UpdateParam(ModelDim, LearningRate, @Beta2.Grad[0], @Beta2.Value[0]);
    // cblas_saxpy(ModelDim, -LearningRate, @Beta2.Grad[0], 1, @Beta2.Value[0], 1);
  end;
end;

// Update Embeddings.
{procedure UpdateEmbeddings(var WModelParams: TWModelParams; var WModelState: TWModelState; const Start: Integer);
begin
  with WModelParams do with WModelState do begin
    UpdateParam(nVocab * ModelDim, LearningRate, @Embeddings.Grad[0,0], @Embeddings.Value[0,0]);

    // Add input-side embedding gradients into Embeddings.Grad.
    for i := 0 to SeqLen - 1 do
      AddScaled(ModelDim, 1.0, @WModelState.StateBlock[0].X.Grad[i,0], @WModelParams.Embeddings.Grad[TokenID[Start + i], 0]);
      // cblas_saxpy(ModelDim, 1.0, @WModelState.X.Grad[i,0], 1, @Embeddings.Grad[v,0], 1);
  end;
end;}

procedure UpdateEmbeddings(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens: TIDimVector);
var
  i, tok: Integer;
begin
  // Add input-side embedding gradients.
  for i := 0 to SeqLen - 1 do begin
    tok := InputTokens[i];

    AddScaled(
      ModelDim,
      1.0,
      @WModelState.StateBlock[0].X.Grad[i,0],
      @WModelParams.Embeddings.Grad[tok,0]
    );
  end;

  // Apply total embedding gradient:
  // output-side tied gradient + input-side gathered gradient.
  UpdateParam(
    nVocab * ModelDim,
    LearningRate,
    @WModelParams.Embeddings.Grad[0,0],
    @WModelParams.Embeddings.Value[0,0]
  );


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

procedure SoftmaxForwardN(const x: PSingle; y: PSingle; const N: Integer);
var
  i: Integer;
  MaxVal, SumVal, InvT: Single;
begin
  if N <= 0 then Exit;

  InvT := 1.0 / Temperature;

  // Find max for numerical stability.
  MaxVal := x[0] * InvT;
  for i := 1 to N - 1 do
    if (x[i] * InvT) > MaxVal then
      MaxVal := x[i] * InvT;

  // Exp and sum.
  SumVal := 0.0;
  for i := 0 to N - 1 do begin
    y[i] := Exp((x[i] * InvT) - MaxVal);
    SumVal := SumVal + y[i];
  end;

  // Normalize.
  if SumVal <> 0.0 then begin
    SumVal := 1.0 / SumVal;
    for i := 0 to N - 1 do
      y[i] := y[i] * SumVal;
  end;
end;

// Softmax procedure forward.
{procedure SoftmaxForward(const x: TFVector; out y: array of Single);
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
end;}

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

