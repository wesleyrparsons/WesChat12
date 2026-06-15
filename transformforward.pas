unit TransformForward;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

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

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk, Start: Integer);

implementation

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk, Start: Integer);
// Run the transformer forward.
var
  h, HeadOffset: Integer;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    if Training then begin
      // Seed random number generation.
      ADropoutSeed := GlobalSeed + UInt64(Start) * 10000 + UInt64(Blk) * 1000 + 1;
      MLPDropoutSeed := GlobalSeed + UInt64(Start) * 10000 + UInt64(Blk) * 1000 + 2;
      RDropoutSeed := GlobalSeed + UInt64(Start) * 10000 + UInt64(Blk) * 1000 + 3;
    end;

    // Display X.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@X.Value[0, 0], X.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Value in transform, before any action.', X.Value, G);
    end;

    // 1. FORWARD: ATTENTION.
    Stage := Blk * 2 + 2;
    // 1A. Layer-Norm. Obtain X1 from X.
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ',  Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '1A. Transform Forward, Layer Norm X');
    // Layer Norm Forward: Input X. Output X1.
    // Obtain input X from Tokenizer for Transformer.
    // Equation: X1 = LayerNorm(X). X, X1 in R^{L × D}. Gamma1, Beta1 in R^{D}.
    // LayerNormForward(X.Value, X1.Value, SeqLen, Gamma1.Value, Beta1.Value, LNXhat1, LNInvStd1);
    LaunchLayerNormForward(X.dValue, X1.dValue, Gamma1.dValue, Beta1.dValue, dLNXhat1, dLNInvStd1, SeqLen, ModelDim);

    {CheckCudaError('LayerNormForward X -> X1'); ```

    cudaMemcpy(@X.Value[0,0], X.dValue, XSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(@X1.Value[0,0], X1.dValue, XSize, cudaMemcpyDeviceToHost);

    Writeln('Block ', Blk, ' X row 0 before LN:');
    for h := 0 to 15 do
      Write(X.Value[0,h]:12:6);
    Writeln;

    Writeln('Block ', Blk, ' X1 row 0 after LN:');
    for h := 0 to 15 do
      Write(X1.Value[0,h]:12:6);
    Writeln;
    Pause;}

    // Display X1.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@X1.Value[0, 0], X1.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Value after layer-norming.', X1.Value, B);
    end;

    // 1B. Split. Split X1 into X1 and accumulate into X4 (implicit).

    // 1C. Multiplication/Overwrite. Obtain Q, K, V from X1.
    if DisplaySubstage then Writeln('' : Stage, '1B-C. Transform Forward 1B, implicit split X1 into X1 and accumulate into X4, and Obtain Q, K, V from X1');

    // Full Size Multiplication/Overwrite: Input X1, Wq. Output Q.
    // Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // MatMulNN(@X1.Value[0, 0], @Wq.Value[0, 0], @Q.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // Writeln('Before Q = x1*Wq');
    CuMatMulNN(cuHandle, X1.dValue, Wq.dValue, Q.dValue, SeqLen, ModelDim, ModelDim);

    // Display Q.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@Q.Value[0, 0], Q.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Q in transform.', Q.Value, G);
    end;

    // Full Size Multiplication/Overwrite: Input X1, Wk. Output K.
    // Equation: K = X1 · Wk. K in R^{L x D}. X1 in R^{L · D}. Wk in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // MatMulNN(@X1.Value[0, 0], @Wk.Value[0, 0], @K.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // Writeln('Before K = x1*Wk');
    CuMatMulNN(cuHandle, X1.dValue, Wk.dValue, K.dValue, SeqLen, ModelDim, ModelDim);

    // Display K.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@K.Value[0, 0], K.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display K, end, in transform.', K.Value, E);
    end;

    // Full Size Multiplication/Overwrite: Input X1, Wv. Output V.
    // Equation: V = X1 · Wv. V in R^{L x D}. X1 in R^{L · D}. Wv in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    // MatMulNN(@X1.Value[0, 0], @Wv.Value[0, 0], @V.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // Writeln('Before V = x1*Wv');
    CuMatMulNN(cuHandle, X1.dValue, Wv.dValue, V.dValue, SeqLen, ModelDim, ModelDim);

    // 1D. RoPE.
    // Q and K were copied from cublas above.
    // ApplyRoPE(Q.Value, WModelState.InvFreq, SeqLen, ModelDim);
    // ApplyRoPE(K.Value, WModelState.InvFreq, SeqLen, ModelDim);
    if DisplaySubstage then Writeln('' : Blk * 2 + 2, '1D. Transform Forward 1D, RoPE Q and K');
    LaunchRoPEForward(Q.dValue, WModelState.dInvFreq, SeqLen, ModelDim);
    LaunchRoPEForward(K.dValue, WModelState.dInvFreq, SeqLen, ModelDim);

    if DisplaySubstage then Writeln('' : Stage, '1E-G. Transform Forward, Obtain Scores1, Autoregressive mask, obtain Scores2, Softmax, ADropout');

    // Multihead Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=HeadDim

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1E. Multiplication. Obtain Scores1.
      // Q_h is Q[*, headOffset .. headOffset+H-1]
      // K_h is K[*, headOffset .. headOffset+H-1]
      // Multiply Q_h (L x H) by K_h^T (H x L), and scale by InvSqrtHeadDim.
      // MatMulFullScaledNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0],
      //   SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);
        // Writeln('Before ScoresHead1 QK');
        CuMatMulFullScaledNT(CuHandle, PSingle(Q.dValue) + HeadOffset, PSingle(K.dValue) + HeadOffset, ScoresHead1[h].dValue,
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);

      // 1F-a. Apply autoregressive mask.
      // Masking: Input ScoresHead1. Output ScoresHead1.
      // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
      // ApplyAutoRegressiveMask(ScoresHead1[h].Value, SeqLen);
      LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);

      // 1F-b. Softmax: ScoresHead1 -> ScoresHead2.
      // for i := 0 to SeqLen - 1 do
      //   SoftmaxForwardN(@ScoresHead1[h].Value[i, 0], @ScoresHead2[h].Value[i, 0], SeqLen);
      LaunchSoftmaxForwardN(ScoresHead1[h].dValue, ScoresHead2[h].dValue, SeqLen, SeqLen, Temperature);

      // 1F-c. Attention dropout.
      // Equation: ScoresHead2 = Dropout(ScoresHead2). ScoresHead in R^{L x L}.
      if Training then
        // for i := 0 to SeqLen - 1 do for j := 0 to SeqLen - 1 do
        //    if Random < ADropOut then
        //      ScoresHead2[h].Value[i, j] := 0.0
        //    else
        //      ScoresHead2[h].Value[i, j] := ScoresHead2[h].Value[i, j] / (1.0 - ADropOut);
        LaunchDropout(ScoresHead2[h].dValue, SeqLen * SeqLen, ADropOut, ADropoutSeed + h);

      // 1G. Multiplication/Overwrite. Obtain X2Head from ScoresHead2.
      // Scoring: Input ScoresHead2, VHead. Output: X.
      // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen
      // MatMulFullNN(@ScoresHead2[h].Value[0,0], @V.Value[0, HeadOffset], @X2.Value[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
      // Writeln('Before ScoresHead2, in 1G');
      CuMatMulFullNN(CuHandle, ScoresHead2[h].dValue, PSingle(V.dValue) + HeadOffset, PSingle(X2.dValue) + HeadOffset,
        SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Display ScoresHead[0].Value, Scores1Head2[1].Value, X2.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@ScoresHead2[0].Value[0, 0], ScoresHead2[0].dValue, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead2[0] after softmax, in transform, before any action.', ScoresHead2[0].Value, G);
      cudaMemcpy(@X2.Value[0, 0], X2.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X2, after Softmax, and concatenation.', X2.Value, B);
    end;

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Forward, Obtain X3 by weighting X2 by W0');

    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    // MatMulNN(@X2.Value[0, 0], @W0.Value[0, 0], @X3.Value[0, 0], SeqLen, ModelDim, ModelDim);
    // Writeln('Before X2, 1H');
    CuMatMulNN(CuHandle, X2.dValue, W0.dValue, X3.dValue, SeqLen, ModelDim, ModelDim);

    // Display X3.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@X3.Value[0, 0], X3.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3, in transform.', X3.Value, B);
    end;

    // 1I. Merge. Obtain X4 from X1 and X3.
    if DisplaySubstage then Writeln('': Stage, '1I. Transform Forward, Obtain X4 from X1 and X3');

    // Merge Addition: Input X1, X3. Output X4.
    // Equation: X4 = X1 + X3. X4 in R^{L · D}. X1 in R^{L · D}. X2 in R^(L x D}.
    // MatAdd(X1.Value, X3.Value, X4.Value, SeqLen, ModelDim);
    CuMatAdd(CuHandle, X1.dValue, X3.dValue, X4.dValue, SeqLen, ModelDim);     // No need to transfer X4.

    // Display X4.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@X4.Value[0, 0], X4.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Value, in transform, after residual added to X3.', X4.Value, G);
    end;

    // 1J. Layer-Norm. Obtain X5 from X4. X4 is already out of cublas.
    if DisplaySubstage then Writeln('': Stage, '1J. Transform Forward, Obtain X5 from Layer Norm Forward X4');

    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    // LayerNormForward(X4.Value, X5.Value, SeqLen, Gamma2.Value, Beta2.Value, LNXhat2, LNInvStd2);
    LaunchLayerNormForward(X4.dValue, X5.dValue, Gamma2.dValue, Beta2.dValue, dLNXhat2, dLNInvStd2, SeqLen, ModelDim);

    // Display X5.Value matrix.
    if VeryVerboseTransform then begin
      cudaMemcpy(@X5.Value[0, 0], X5.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X5.Value, in transform, before FFN.', X5.Value, G);
    end;

      // 2. FORWARD FFN.
      Stage := Stage + 2;
      // 2A. Multiplication/Overwrite. Obtain Hidden1 from X5 and W1.
      if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ',  Transform Forward');
      if DisplaySubstage then Writeln('': Stage, '2A. Transform Forward, Obtain Hidden1 from X5 and W1');

      // Expansion: Input X5, W1. Output Hidden1.
      // Equation: Hidden1 = X5 · W1. Hidden1 in R^{L x DB}. X5 in R^{L x D}. W1 in R^{D x DB}.
      // MatMulNN(@X5.Value[0, 0], @W1.Value[0, 0], @Hidden1.Value[0, 0], SeqLen, ModelDimProj, ModelDim);
      CuMatMulNN(CuHandle, X5.dValue, W1.dValue, Hidden1.dValue, SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden1 from Hidden1 and b1.
      if DisplaySubstage then Writeln('': Stage, '2B. Transform Forward, Obtain Hidden1 from Hidden1 and b1');

      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 + b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      // AddMatVec(@Hidden1.Value, b1.Value, SeqLen, ModelDimProj);
      // for i := 0 to SeqLen - 1 do
        // AddScaled(ModelDimProj, 1.0, @b1.Value[0], @Hidden1.Value[i,0]);
      LaunchAddBiasRows(Hidden1.dValue, b1.dValue, SeqLen, ModelDimProj);

      // Display Hidden1.Value matrix.
      if VeryVerboseTransform then begin
        cudaMemcpy(@Hidden1.Value[0, 0], Hidden1.dValue, HiddenSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display Hidden1.Value, in transform,  after adding b1, and before ReLU.', Hidden1.Value, G);
      end;

      // 2C. ReLU. Obtain Hidden2 from Hidden1.
      if DisplaySubstage then Writeln('': Stage, '2C. Transform Forward, Obtain Hidden2 by ReLU Forward from Hidden1 and MLP Dropout');

      // Activation: Input Hidden1. Output Hidden2.
      // Equation: Hidden2 = ReLU(Hidden1).
      // ReLUMaskForward(Hidden1.Value, Hidden2.Value)
      LaunchReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);

      // Do MLP dropout.
      if Training then
        // for i := 0 to SeqLen - 1 do for j := 0 to ModelDimProj - 1 do
        //    if Random < MLPDropout then
        //      Hidden2.Value[i, j] := 0.0
        //    else
        //      Hidden2.Value[i, j] := Hidden2.Value[i, j] / (1.0 - MLPDropOut);
        LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);

      // 2D. Multiplication/Overwrite. Obtain X6 from Hidden2.
      if DisplaySubstage then Writeln('': Stage, '2D. Transform Forward, Obtain X6 from Hidden2');

      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      // MatMulNN(@Hidden2.Value[0, 0], @W2.Value[0, 0], @X6.Value[0, 0], SeqLen, ModelDim, ModelDimProj);
      CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from X6 and b2.
      if DisplaySubstage then Writeln('': Stage, '2E. Transform Forward, Obtain X6 from X6 and b2');

      // Addition: Input X6, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{D}.
      // AddMatVec(@X6.Value, @b2.Value, SeqLen, ModelDim);
      // for i := 0 to SeqLen - 1 do
        // AddScaled(ModelDim, 1.0, @b2.Value[0], @X6.Value[i,0]);
      LaunchAddBiasRows(X6.dValue, b2.dValue, SeqLen, ModelDim);

      // Display X6.Value matrix.
      if VeryVerboseTransform then begin
        cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X6, in transform, after contraction.', X6.Value, B);
      end;

      // 2F. Addition/Merge and residual dropout. Obtain X7 from X5 and X6.
      if DisplaySubstage then Writeln('': Stage, '2F. Transform Forward, Obtain X7 from X5 and X6, and RDropout');

      // Do residual dropout.
      if Training then
        // for i := 0 to SeqLen - 1 do for j := 0 to ModelDim - 1 do
        //    if Random < RDropout then
        //      X6.Value[i, j] := 0.0
        //    else
        //      X6.Value[i, j] := X6.Value[i, j] / (1.0 - RDropOut);
        LaunchDropout(X6.dValue, SeqLen * ModelDim, RDropout, RDropoutSeed);

      // Residual merge: Input X5, X6. Output X7.
      // Equation: X7 = X5 + X6. X7 in R^{L · D}. X5 in R^{L · D}. X6 in R^{L x D}.
      // MatAdd(X5.Value, X6.Value, X7.Value, SeqLen, ModelDim);
      CuMatAdd(CuHandle, X5.dValue, X6.dValue, X7.dValue, SeqLen, ModelDim);

      // Display X7.Value matrix.
      if VeryVerboseTransform then begin
        cudaMemcpy(@X7.Value[0, 0], X7.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X7.Value, in transform, after residual added to X6.', X7.Value, B);
      end;

  end;   // End with WModel.
end;     // End RunTransformForward.

end.

