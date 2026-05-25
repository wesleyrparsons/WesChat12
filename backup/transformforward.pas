unit TransformForward;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Matrix,
  SysUtils,
  Util;

const
  InvSqrtHeadDim: Single = 1 / Sqrt(HeadDim);         // Used in softmax.
  RowMajor = 101;       // Row Major.
  NoTrans  = 111;       // No transposition.
  Trans    = 112;       // Transposition.

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);

implementation

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);
// Run the transformer forward.
var
  h, i, j, HeadOffset: Integer;
  Seed: UInt64;
begin
  // Seed RNG.
  Seed := GetTickCount64;

  // Display entry to transform.
  writeln('Entering Forward Transformer');

  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin
  // Display X.Value matrix.
  VTPDisplayX('Display X.Value in transform, before any action.', X.Value, G);

  // 1. FORWARD STAGE: ATTENTION.

    // 1A. Layer-Norm. Obtain X1 from X.
    Writeln('          Transform Forward Stage 1A');
    // Layer Norm: Input X. Output X1.
    // Obtain input X from Tokenizer for Transformer stage.
    // Purpose: Normalization.
    // Equation: X1 = LayerNorm(X). X, X1 in R^{L × D}. Gamma1, Beta1 in R^{D}.
    // cblas.
    // LayerNormForward(X.Value, X1.Value, SeqLen, Gamma1.Value, Beta1.Value, LNXhat1, LNInvStd1);
    // cuda kernel. No need to copy X1.dValue, or dLNXHat1 or dLNInvStd1; they ar outputs.
    cudaMemcpy(X.dValue, @X.Value[0, 0], XSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Gamma1.dValue, @Gamma1.Value[0], ModelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Beta1.dValue, @Beta1.Value[0], ModelSize, cudaMemcpyHostToDevice);
    LaunchLayerNormForward(X.dValue, X1.dValue, Gamma1.dValue, Beta1.dValue, dLNXhat1, dLNInvStd1, SeqLen, ModelDim);

    // Display X1.Value matrix.
    cudaMemcpy(@X1.Value[0, 0], X1.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display X1.Value after layer-norming.', X1.Value, B);

    // 1B. Split. Implicit split into X1 and accumulate into X4.
    Writeln('          Transform Forward Stage 1B (Implicit)');

    // 1C. Multiplication/Overwrite. Obtain Q, K, V from X1.
    Writeln('          Transform Forward Stage 1C');

    // Full Size Multiplication/Overwrite: Input X1, Wq. Output Q.
    // Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // cblas.
    // MatMulNN(@X1.Value[0, 0], @Wq.Value[0, 0], @Q.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // cublas.
    cudaMemcpy(Q.dValue, @Q.Value[0, 0], XSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Wq.dValue, @Wq.Value[0, 0], WeightSize, cudaMemcpyHostToDevice);
    CuMatMulNN(cuHandle, X1.dValue, Wq.dValue, Q.dValue, SeqLen, ModelDim, ModelDim);

    // Display Q.Value matrix.
    cudaMemcpy(@Q.Value[0, 0], Q.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display Q in transform.', Q.Value, G);

    // Full Size Multiplication/Overwrite: Input X1, Wk. Output K.
    // Equation: K = X1 · Wk. K in R^{L x D}. X1 in R^{L · D}. Wk in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // cblas.
    // MatMulNN(@X1.Value[0, 0], @Wk.Value[0, 0], @K.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // cublas.
    cudaMemcpy(K.dValue, @K.Value[0, 0], XSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Wk.dValue, @Wk.Value[0, 0], WeightSize, cudaMemcpyHostToDevice);
    CuMatMulNN(cuHandle, X1.dValue, Wk.dValue, K.dValue, SeqLen, ModelDim, ModelDim);

    // Display K.Value matrix.
    cudaMemcpy(@K.Value[0, 0], K.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display K, end, in transform.', K.Value, E);

    // Full Size Multiplication/Overwrite: Input X1, Wv. Output V.
    // Equation: V = X1 · Wv. V in R^{L x D}. X1 in R^{L · D}. Wv in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // cblas.
    // MatMulNN(@X1.Value[0, 0], @Wv.Value[0, 0], @V.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // cublas.
    cudaMemcpy(V.dValue, @V.Value[0, 0], XSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Wv.dValue, @Wv.Value[0, 0], WeightSize, cudaMemcpyHostToDevice);     // No need to copy V.
    CuMatMulNN(cuHandle, X1.dValue, Wv.dValue, V.dValue, SeqLen, ModelDim, ModelDim);

    // 1D. RoPE.
    // Q and K were copied from cublas above.
    // cblas.
    // ApplyRoPE(Q.Value, WModelState.InvFreq, SeqLen, ModelDim);
    // ApplyRoPE(K.Value, WModelState.InvFreq, SeqLen, ModelDim);
    // cuda kernel.
    LaunchRoPEForward(Q.dValue, WModelState.dInvFreq, SeqLen, ModelDim);
    LaunchRoPEForward(K.dValue, WModelState.dInvFreq, SeqLen, ModelDim);

    // 1E. Multiplication. Obtain Scores1.
    Writeln('          Transform Forward Stage 1E');

    // Multihead Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=HeadDim

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // Q_h is Q[*, headOffset .. headOffset+H-1]
      // K_h is K[*, headOffset .. headOffset+H-1]
      // Multiply Q_h (L x H) by K_h^T (H x L), and scale by InvSqrtHeadDim.
      // cblas.
      // MatMulFullScaledNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0],
        // SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);
      {OR THIS WAY:
      cblas_sgemm(RowMajor, NoTrans, Trans, SeqLen, SeqLen, HeadDim, InvSqrtHeadDim, @Q.Value[0, HeadOffset],
        ModelDim, @K.Value[0, HeadOffset], ModelDim, 0.0, @ScoresHead1[h].Value[0, 0], SeqLen);
      OR THIS WAY MatMulNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0], SeqLen, SeqLen, HeadDim);
      And also scale by InvSqrtHeadDim}
      // cublas.
      CuMatMulFullScaledNT(CuHandle, PSingle(Q.dValue) + HeadOffset, PSingle(K.dValue) + HeadOffset, ScoresHead1[h].dValue,
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);
    end;

    // Display ScoresHead[0].Value matrix. Copy all [h] to cuda even tho displaying [1].
    for h := 0 to nHead - 1 do
      cudaMemcpy(@ScoresHead1[h].Value[0, 0], ScoresHead1[h].dValue, ScoresSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display ScoresHead1[1] before standardizing.', ScoresHead1[1].Value, B);

    // 1F. Mask & Softmax & Dropout. Obtain Scores2. All not done in cublas.
    Writeln('          Transform Forward Stage 1F');

    // Masking: Input ScoresHead1. Output ScoresHead1.
    // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
    // cblas.
    // for h := 0 to nHead - 1 do
    // ApplyAutoRegressiveMask(ScoresHead1[h].Value, SeqLen);
    // cuda kernel.
    for h := 0 to nHead - 1 do
      // Non-cuda kernel.
      // LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);
      // cuda kernel.
      LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);

    // Softmax: Input ScoresHead1. Output ScoresHead2.
    // Equation: ScoresHead2 = Softmax(ScoresHead1). ScoresHead in R^{L x L}.
    for h := 0 to nHead - 1 do
      // Non-cuda kernel.
      // for i := 0 to SeqLen - 1 do
      //   SoftmaxForwardN(@ScoresHead1[h].Value[i, 0], @ScoresHead2[h].Value[i, 0], SeqLen);
      // cuda kernel.
      LaunchSoftmaxForwardN(ScoresHead1[h].dValue, ScoresHead2[h].dValue, SeqLen, SeqLen, Temperature);

    // Display Scores1Head2[1].Value matrix. Copy all [h] to cuda even tho displaying [1].
    for h := 0 to nHead - 1 do
      cudaMemcpy(@ScoresHead2[h].Value[0, 0], ScoresHead2[h].dValue, ScoresSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display ScoresHead2[1] after softmax, in transform, before any action.', ScoresHead2[1].Value, G);

    // Do attention dropout.
    // Equation: ScoresHead2 = Dropout(ScoresHead2). ScoresHead in R^{L x L}.
    if Training then
      for h := 0 to nHead - 1 do
        // Non-cuda kernel.
        // for i := 0 to SeqLen - 1 do for j := 0 to SeqLen - 1 do
        //    if Random < ADropOut then
        //      ScoresHead2[h].Value[i, j] := 0.0
        //    else
        //      ScoresHead2[h].Value[i, j] := ScoresHead2[h].Value[i, j] / (1.0 - ADropOut);
        // cuda kernel.
        LaunchDropout(ScoresHead2[h].dValue, SeqLen * SeqLen, ADropOut, UInt64(Seed) + h);

    // 1G. Multiplication/Overwrite. Obtain X2Head from ScoresHead2.
    Writeln('          Transform Forward Stage 1G');

    // Scoring: Input ScoresHead2, VHead. Output: X.
    // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      // cblas.
      // MatMulFullNN(@ScoresHead2[h].Value[0,0], @V.Value[0, HeadOffset], @X2.Value[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
      // cublas.
      cudaMemcpy(ScoresHead2[h].dValue, @ScoresHead2[h].Value[0, 0], ScoresSize, cudaMemcpyHostToDevice);
      CuMatMulFullNN(CuHandle, ScoresHead2[h].dValue, PSingle(V.dValue) + HeadOffset, PSingle(X2.dValue) + HeadOffset,
        SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Display X2.Value matrix.
    cudaMemcpy(@X2.Value[0, 0], X2.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display X2, after Softmax, and concatenation.', X2.Value, B);

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.
    Writeln('          Transform Forward Stage 1H');

    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    // cblas.
    // MatMulNN(@X2.Value[0, 0], @W0.Value[0, 0], @X3.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // cublas.
    cudaMemcpy(W0.dValue, @W0.Value[0, 0], WeightSize, cudaMemcpyHostToDevice);
    CuMatMulNN(CuHandle, X2.dValue, W0.dValue, X3.dValue, SeqLen, ModelDim, ModelDim);

    // Display X3.Value matrix.
    cudaMemcpy(@X3.Value[0, 0], X3.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display X3, in transform.', X3.Value, B);

    // 1I. Merge. Obtain X4 from X1 and X3.
    Writeln('          Transform Forward Stage 1I');

    // Merge Addition: Input X1, X3. Output X4.
    // Equation: X4 = X1 + X3. X4 in R^{L · D}. X1 in R^{L · D}. X2 in R^(L x D}.
    // cblas.
    // MatAdd(X1.Value, X3.Value, X4.Value, SeqLen, ModelDim);
    // cublas.
    CuMatAdd(CuHandle, X1.dValue, X3.dValue, X4.dValue, SeqLen, ModelDim);     // No need to transfer X4.

    // Display X4.Value matrix.
    cudaMemcpy(@X4.Value[0, 0], X4.dValue, XSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display X4.Value, in transform, after residual added to X3.', X4.Value, G);

    // 1J. Layer-Norm. Obtain X5 from X4. X4 is already out of cublas.
    Writeln('          Transform Forward Stage 1J');

    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    // non-cuda kernel.
    // LayerNormForward(X4.Value, X5.Value, SeqLen, Gamma2.Value, Beta2.Value, LNXhat2, LNInvStd2);
    // cuda kernel.
    LaunchLayerNormForward(X4.dValue, X5.dValue, Gamma1.dValue, Beta1.dValue, dLNXhat1, dLNInvStd1, SeqLen, ModelDim);

    // Display X5.Value matrix.
    VTPDisplayX('Display X5.Value, in transform, before FFN.', X5.Value, G);

      // 2. STAGE FORWARD FFN.

      // 2A. Multiplication/Overwrite. Obtain Hidden1 from X5 and W1.
      Writeln('            Transform Forward Stage 2A');

      // Expansion: Input X5, W1. Output Hidden1.
      // Equation: Hidden1 = X5 · W1. Hidden1 in R^{L x DB}. X5 in R^{L x D}. W1 in R^{D x DB}.
      // cblas
      // MatMulNN(@X5.Value[0, 0], @W1.Value[0, 0], @Hidden1.Value[0, 0], SeqLen, ModelDimProj, ModelDim);
      // cublas.
      cudaMemcpy(W1.dValue, @W1.Value[0, 0], WeightProjSize, cudaMemcpyHostToDevice);
      cudaMemcpy(X5.dValue, @X5.Value[0, 0], XSize, cudaMemcpyHostToDevice);         // No need to copy Hidden1.
      CuMatMulNN(CuHandle, X5.dValue, W1.dValue, Hidden1.dValue, SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden1 from Hidden1 and b1.
      Writeln('            Transform Forward Stage 2B');

      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 * b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      // cblas.
      // AddMatVec(@Hidden1.Value, b1.Value, SeqLen, ModelDimProj);
      //cublas.
      cudaMemcpy(b1.dValue, @b1.Value[0], bProjSize, cudaMemcpyHostToDevice);
      for i := 0 to SeqLen - 1 do
        // cblas.
        // AddScaled(ModelDimProj, 1.0, @b1.Value[0], @Hidden1.Value[i,0]);
        // cublas.
        CuAddScaled(CuHandle, ModelDimProj, 1.0, b1.dValue, PSingle(Hidden1.dValue) + i * ModelDimProj);

      // Display Hidden1.Value matrix.
      cudaMemcpy(@Hidden1.Value[0, 0], Hidden1.dValue, HiddenSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Hidden1.Value, in transform,  after adding b1, and before ReLU.', Hidden1.Value, G);

      // 2C. ReLU. Obtain Hidden2 from Hidden1.
      Writeln('            Transform Forward Stage 2C');

      // Activation: Input Hidden1. Output Hidden2.
      // Equation: Hidden2 = ReLU(Hidden1).
      if not WesChatKernelPresent then
        ReLUMaskForward(Hidden1.Value, Hidden2.Value);
      else
        LaunchReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);

      // Do MLP dropout.
      if Training then
        // Non-cuda kernel.
        // for i := 0 to SeqLen - 1 do for j := 0 to ModelDimProj - 1 do
        //    if Random < MLPDropout then
        //      Hidden2.Value[i, j] := 0.0
        //    else
        //      Hidden2.Value[i, j] := Hidden2.Value[i, j] / (1.0 - MLPDropOut);
        // cuda kernel.
        LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, RDropOut, UInt64(GetTickCount64));

      // 2D. Multiplication/Overwrite. Obtain X6 from Hidden2.
      Writeln('            Transform Forward Stage 2D');

      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      // cblas
      // MatMulNN(@Hidden2.Value[0, 0], @W2.Value[0, 0], @X6.Value[0, 0], SeqLen, ModelDim, ModelDimProj);
      // cublas.
      cudaMemcpy(Hidden2.dValue, @Hidden2.Value[0, 0], HiddenSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dValue, @W2.Value[0, 0], WeightProjSize, cudaMemcpyHostToDevice);   // No need to memcpy X6.
      CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from X6 and b2.
      Writeln('            Transform Forward Stage 2E');

      // Addition: Input X6, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{D}.
      // AddMatVec(@X6.Value, @b2.Value, SeqLen, ModelDim);
      cudaMemcpy(b2.dValue, @b2.Value[0], bSize, cudaMemcpyHostToDevice);
      for i := 0 to SeqLen - 1 do
        // cblas.
        // AddScaled(ModelDim, 1.0, @b2.Value[0], @X6.Value[i,0]);
        // cublas.
        CuAddScaled(CuHandle, ModelDim, 1.0, b2.dValue, PSingle(X6.dValue) + i * ModelDim);

      // Display X6.Value matrix.
      cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X6, in transform, after contraction.', X6.Value, B);

      // 2F. Addition/Merge and residual dropout. Obtain X7 from X5 and X6.
      Writeln('            Transform Forward Stage 2F');

      // Do residual dropout.
      if Training then
        // Non-cuda kernel.
        // for i := 0 to SeqLen - 1 do for j := 0 to ModelDim - 1 do
        //    if Random < RDropout then
        //      X6.Value[i, j] := 0.0
        //    else
        //      X6.Value[i, j] := Hidden2.Value[i, j] / (1.0 - RDropOut);
        // cuda kernel.
        LaunchDropout(X3.dValue, SeqLen * ModelDim, RDropout, UInt64(GetTickCount64));

      // Backprop Merge Addition: Input Residual X6, X5. Output X7.
      // Equation: X7 = X5 + X6. X7 in R^{L · D}. X5 in R^{L · D}. X6 in R^{L x D}.
      // cblas.
      // MatAdd(X5.Value, X6.Value, X7.Value, SeqLen, ModelDim);
      // cublas. Already have X5 in cublas.
      cudaMemcpy(X6.dValue, @X6.Value[0, 0], XSize, cudaMemcpyHostToDevice);
      CuMatAdd(CuHandle, X5.dValue, X6.dValue, X7.dValue, SeqLen, ModelDim);   // No need to memcpy X7.

      // Display X7.Value matrix.
      cudaMemcpy(@X7.Value[0, 0], X7.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X7.Value, in transform, after residual added to X6.', X7.Value, B);

  end;   // End with WModel.
end;     // End RunTransformForward.

end.

