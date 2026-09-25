unit TransformForward;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  Display,
  Global,
  Matrix,
  SysUtils,
  Util;

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

{ Transformer forward pass }

// Run one transformer block forward.
procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h, HeadOffset: Integer;
  StepSeed: UInt64;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    { Dropout seeds }

    // Set deterministic dropout seeds for this training step and block.
    if Training then begin
      StepSeed := GlobalSeed + UInt64(GlobalStep) * 100000 + UInt64(Blk) * 1000;
      ADropoutSeed := StepSeed + 1;
      MLPDropoutSeed := StepSeed + 100;
      RDropoutSeed := StepSeed + 200;
    end;

    // Display the input X matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X.Value[0, 0], X.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Value in transform, before any action.', X.Value, G);
    end;

    { Attention forward pass }

    Stage := Blk * 4 + 2;

    // 1A. LayerNorm: X -> X1.
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ', Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '1A. Transform Forward, LayerNorm X');
    // Equation: X1 = LayerNorm(X). X and X1 are L x D; Gamma1 and Beta1 are D.
    LaunchLayerNormForward(X.dValue, X1.dValue, Gamma1.dValue, Beta1.dValue, dLNXhat1, dLNInvStd1, SeqLen, ModelDim);

    // Display X1 after LayerNorm.
    if VerboseTransform then begin
      cudaMemcpy(@X1.Value[0, 0], X1.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Value after LayerNorm.', X1.Value, B);
    end;

    // 1B-C. Preserve X for the residual path and project X1 into Q, K, and V.
    if DisplaySubstage then Writeln('' : Stage, '1B-C. Transform Forward, project X1 into Q, K, and V');

    // Q = X1 * Wq. Q and X1 are L x D; Wq is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wq.dValue, Q.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@Q.Value[0, 0], Q.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Q in transform.', Q.Value, G);
    end;

    // K = X1 * Wk. K and X1 are L x D; Wk is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wk.dValue, K.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@K.Value[0, 0], K.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display K in transform.', K.Value, E);
    end;

    // V = X1 * Wv. V and X1 are L x D; Wv is D x D.
    CuMatMulNN(CuHandle, X1.dValue, Wv.dValue, V.dValue, SeqLen, ModelDim, ModelDim);

    // 1D. Apply RoPE independently within each attention head to Q and K.
    if DisplaySubstage then Writeln('' : Stage, '1D. Transform Forward, RoPE Q and K');
    LaunchRoPEForward(Q.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEForward(K.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    // 1E-G. Compute attention scores, apply causal masking and softmax, apply attention dropout, and multiply by V.
    if DisplaySubstage then Writeln('' : Stage, '1E-G. Transform Forward, attention scores, mask, softmax, attention dropout, and V projection');

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1E. ScoresHead1 = Qh * Kh^T / sqrt(HeadDim). Qh and Kh are L x H; ScoresHead1 is L x L.
      CuMatMulFullScaledNT(CuHandle, PSingle(Q.dValue) + HeadOffset, PSingle(K.dValue) + HeadOffset, ScoresHead1[h].dValue,
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);

      // 1F-a. Apply the autoregressive mask in place to ScoresHead1.
      LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);

      // 1F-b. Softmax: ScoresHead1 -> ScoresHead2.
      LaunchSoftmaxForward(ScoresHead1[h].dValue, ScoresHead2[h].dValue, SeqLen, SeqLen, 1.0);

      // 1F-c. During training, preserve the pre-dropout softmax probabilities in ScoresHead1, then apply attention dropout to ScoresHead2.
      if Training then begin
        cudaMemcpy(ScoresHead1[h].dValue, ScoresHead2[h].dValue, ScoresSize, cudaMemcpyDeviceToDevice);
        LaunchDropout(ScoresHead2[h].dValue, SeqLen * SeqLen, ADropOut, ADropoutSeed + h);
      end;

      // 1G. X2h = ScoresHead2 * Vh. ScoresHead2 is L x L; Vh is L x H; X2h is L x H.
      CuMatMulFullNN(CuHandle, ScoresHead2[h].dValue, PSingle(V.dValue) + HeadOffset, PSingle(X2.dValue) + HeadOffset,
        SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Display attention probabilities for head 0 and the concatenated X2 matrix.
    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead2[0].Value[0, 0], ScoresHead2[0].dValue, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead2[0] after softmax and attention dropout.', ScoresHead2[0].Value, G);
      cudaMemcpy(@X2.Value[0, 0], X2.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X2 after attention and head concatenation.', X2.Value, B);
    end;

    // 1H. Project the concatenated attention output through W0: X2 -> X3.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Forward, obtain X3 by weighting X2 by W0');
    // Equation: X3 = X2 * W0. X2 and X3 are L x D; W0 is D x D.
    CuMatMulNN(CuHandle, X2.dValue, W0.dValue, X3.dValue, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X3.Value[0, 0], X3.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3 in transform.', X3.Value, B);
    end;

    // 1I. Attention residual merge: X4 = X + X3.
    if DisplaySubstage then Writeln('' : Stage, '1I. Transform Forward, obtain X4 from X and X3');
    CuMatAdd(CuHandle, X.dValue, X3.dValue, X4.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X4.Value[0, 0], X4.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Value after adding the attention residual.', X4.Value, G);
    end;

    // 1J. LayerNorm: X4 -> X5.
    if DisplaySubstage then Writeln('' : Stage, '1J. Transform Forward, obtain X5 from LayerNorm X4');
    // Equation: X5 = LayerNorm(X4). X4 and X5 are L x D; Gamma2 and Beta2 are D.
    LaunchLayerNormForward(X4.dValue, X5.dValue, Gamma2.dValue, Beta2.dValue, dLNXhat2, dLNInvStd2, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X5.Value[0, 0], X5.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X5.Value before the feed-forward network.', X5.Value, G);
    end;

    { Feed-forward forward pass }

    Stage := Stage + 2;

    // 2A. Expand X5 through W1: X5 -> Hidden1.
    if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ', Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '2A. Transform Forward, obtain Hidden1 from X5 and W1');
    // Equation: Hidden1 = X5 * W1. X5 is L x D; W1 is D x DB; Hidden1 is L x DB.
    CuMatMulNN(CuHandle, X5.dValue, W1.dValue, Hidden1.dValue, SeqLen, ModelDimProj, ModelDim);

    // 2B. Add b1 to every row of Hidden1.
    if DisplaySubstage then Writeln('' : Stage, '2B. Transform Forward, add b1 to Hidden1');
    LaunchAddBiasRows(Hidden1.dValue, b1.dValue, SeqLen, ModelDimProj);

    if VerboseTransform then begin
      cudaMemcpy(@Hidden1.Value[0, 0], Hidden1.dValue, HiddenSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Hidden1.Value after adding b1 and before ReLU.', Hidden1.Value, G);
    end;

    // 2C. ReLU activation: Hidden1 -> Hidden2, followed by MLP dropout during training.
    if DisplaySubstage then Writeln('' : Stage, '2C. Transform Forward, ReLU Hidden1 into Hidden2 and apply MLP dropout');
    LaunchReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);

    if Training then
      LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);

    // 2D. Contract Hidden2 through W2: Hidden2 -> X6.
    if DisplaySubstage then Writeln('' : Stage, '2D. Transform Forward, obtain X6 from Hidden2 and W2');
    // Equation: X6 = Hidden2 * W2. Hidden2 is L x DB; W2 is DB x D; X6 is L x D.
    CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);

    // 2E. Add b2 to every row of X6.
    if DisplaySubstage then Writeln('' : Stage, '2E. Transform Forward, add b2 to X6');
    LaunchAddBiasRows(X6.dValue, b2.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X6 in transform after contraction and b2.', X6.Value, B);
    end;

    // 2F. Apply residual dropout to X6 during training, then merge with X4: X7 = X4 + X6.
    if DisplaySubstage then Writeln('' : Stage, '2F. Transform Forward, residual dropout and obtain X7 from X4 and X6');

    if Training then
      LaunchDropout(X6.dValue, SeqLen * ModelDim, RDropout, RDropoutSeed);

    CuMatAdd(CuHandle, X4.dValue, X6.dValue, X7.dValue, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X7.Value[0, 0], X7.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X7.Value after the feed-forward residual merge.', X7.Value, B);
    end;

  end;
end;

end.

{unit TransformForward;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  Display,
  Global,
  Matrix,
  SysUtils,
  Util;

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
// Run the transformer forward.
var
  h, HeadOffset: Integer;
  StepSeed: UInt64;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    // Seed random number generation.
    if Training then begin
      StepSeed := GlobalSeed + UInt64(GlobalStep) * 100000 + UInt64(Blk) * 1000;
      ADropoutSeed := StepSeed + 1;
      MLPDropoutSeed := StepSeed + 100;
      RDropoutSeed := StepSeed + 200;
    end;

    // Display X.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X.Value[0, 0], X.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Value in transform, before any action.', X.Value, G);
    end;

    // 1. FORWARD: ATTENTION.
    Stage := Blk * 4 + 2;
    // 1A. Layer-Norm. Obtain X1 from X.
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ',  Transform Forward');
    if DisplaySubstage then Writeln('' : Stage, '1A. Transform Forward, Layer Norm X');
    // Layer Norm Forward: Input X. Output X1.
    // Equation: X1 = LayerNorm(X). X, X1 in R^{L × D}. Gamma1, Beta1 in R^{D}.
    LaunchLayerNormForward(X.dValue, X1.dValue, Gamma1.dValue, Beta1.dValue, dLNXhat1, dLNInvStd1, SeqLen, ModelDim);

    // Display X1.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X1.Value[0, 0], X1.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Value after layer-norming.', X1.Value, B);
    end;

    // 1B. Residual path preserves X for the later merge X4 = X + X3.
    // 1C. Multiplication/Overwrite. Obtain Q, K, V from X1.
    if DisplaySubstage then Writeln('' : Stage, '1B-C. Transform Forward, project X1 into Q, K, and V');

    // Full Size Multiplication/Overwrite: Input X1, Wq. Output Q.
    // Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    CuMatMulNN(cuHandle, X1.dValue, Wq.dValue, Q.dValue, SeqLen, ModelDim, ModelDim);

    // Display Q.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@Q.Value[0, 0], Q.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Q in transform.', Q.Value, G);
    end;

    // Full Size Multiplication/Overwrite: Input X1, Wk. Output K.
    // Equation: K = X1 · Wk. K in R^{L x D}. X1 in R^{L · D}. Wk in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    CuMatMulNN(cuHandle, X1.dValue, Wk.dValue, K.dValue, SeqLen, ModelDim, ModelDim);

    // Display K.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@K.Value[0, 0], K.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display K, end, in transform.', K.Value, E);
    end;

    // Full Size Multiplication/Overwrite: Input X1, Wv. Output V.
    // Equation: V = X1 · Wv. V in R^{L x D}. X1 in R^{L · D}. Wv in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
    CuMatMulNN(cuHandle, X1.dValue, Wv.dValue, V.dValue, SeqLen, ModelDim, ModelDim);

    // 1D. RoPE.
    // ApplyRoPE(Q.Value, WModelState.InvFreq, SeqLen, ModelDim);
    // ApplyRoPE(K.Value, WModelState.InvFreq, SeqLen, ModelDim);
    if DisplaySubstage then Writeln('' : Stage, '1D. Transform Forward 1D, RoPE Q and K');
    LaunchRoPEForward(Q.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEForward(K.dValue, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    if DisplaySubstage then Writeln('' : Stage, '1E-G. Transform Forward, Obtain Scores1, Autoregressive mask, obtain Scores2, Softmax, ADropout');

    // Multihead Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1. That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scorec M=SeqLen N=SeqLen K=HeadDim
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1E. Multiplication. Obtain Scores1.
      // Q_h is Q[*, headOffset .. headOffset+H-1]
      // K_h is K[*, headOffset .. headOffset+H-1]
      // Multiply Q_h (L x H) by K_h^T (H x L), and scale by InvSqrtHeadDim.
      CuMatMulFullScaledNT(CuHandle, PSingle(Q.dValue) + HeadOffset, PSingle(K.dValue) + HeadOffset, ScoresHead1[h].dValue,
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);

      // 1F-a. Apply autoregressive mask.
      // Masking: Input ScoresHead1. Output ScoresHead1.
      // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
      LaunchAutoRegressiveMask(ScoresHead1[h].dValue, SeqLen);

      // 1F-b. Softmax: ScoresHead1 -> ScoresHead2.
      // for i := 0 to SeqLen - 1 do
      LaunchSoftmaxForward(ScoresHead1[h].dValue, ScoresHead2[h].dValue, SeqLen, SeqLen, 1.0);

      // 1F-c. Attention dropout.
      // Equation: ScoresHead2 = Dropout(ScoresHead2). ScoresHead in R^{L x L}.
      // During training, preserve the pre-dropout softmax probabilities.
      // The masked logits in ScoresHead1 are no longer needed.
      if Training then begin
        cudaMemcpy(ScoresHead1[h].dValue, ScoresHead2[h].dValue, ScoresSize, cudaMemcpyDeviceToDevice);
        LaunchDropout(ScoresHead2[h].dValue, SeqLen * SeqLen, ADropOut, ADropoutSeed + h);
      end;

      // 1G. Multiplication/Overwrite. Obtain X2Head from ScoresHead2.
      // Scoring: Input ScoresHead2, VHead. Output: X.
      // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen
      CuMatMulFullNN(CuHandle, ScoresHead2[h].dValue, PSingle(V.dValue) + HeadOffset, PSingle(X2.dValue) + HeadOffset,
        SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;      // nHead loop.

    // Display ScoresHead[0].Value, Scores1Head2[1].Value, X2.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead2[0].Value[0, 0], ScoresHead2[0].dValue, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead2[0] after softmax, in transform, before any action.', ScoresHead2[0].Value, G);
      cudaMemcpy(@X2.Value[0, 0], X2.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X2, after Softmax, and concatenation.', X2.Value, B);
    end;

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Forward, Obtain X3 by weighting X2 by W0');

    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    CuMatMulNN(CuHandle, X2.dValue, W0.dValue, X3.dValue, SeqLen, ModelDim, ModelDim);

    // Display X3.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X3.Value[0, 0], X3.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3, in transform.', X3.Value, B);
    end;

    // 1I. Merge. Obtain X4 from X and X3.
    if DisplaySubstage then Writeln('': Stage, '1I. Transform Forward, Obtain X4 from X and X3');

    // Merge Addition: Input X, X3. Output X4.
    // Equation: X4 = X + X3. X4 in R^{L · D}. X in R^{L · D}. X2 in R^{L x D}.
    CuMatAdd(CuHandle, X.dValue, X3.dValue, X4.dValue, SeqLen, ModelDim);     // No need to transfer X4.

    // Display X4.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X4.Value[0, 0], X4.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Value, in transform, after residual added to X3.', X4.Value, G);
    end;

    // 1J. Layer-Norm. Obtain X5 from X4. X4 is already out of cublas.
    if DisplaySubstage then Writeln('': Stage, '1J. Transform Forward, Obtain X5 from Layer Norm Forward X4');

    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    LaunchLayerNormForward(X4.dValue, X5.dValue, Gamma2.dValue, Beta2.dValue, dLNXhat2, dLNInvStd2, SeqLen, ModelDim);

    // Display X5.Value matrix.
    if VerboseTransform then begin
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
      CuMatMulNN(CuHandle, X5.dValue, W1.dValue, Hidden1.dValue, SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden1 from Hidden1 and b1.
      if DisplaySubstage then Writeln('': Stage, '2B. Transform Forward, Obtain Hidden1 from Hidden1 and b1');

      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 + b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      LaunchAddBiasRows(Hidden1.dValue, b1.dValue, SeqLen, ModelDimProj);

      // Display Hidden1.Value matrix.
      if VerboseTransform then begin
        cudaMemcpy(@Hidden1.Value[0, 0], Hidden1.dValue, HiddenSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display Hidden1.Value, in transform,  after adding b1, and before ReLU.', Hidden1.Value, G);
      end;

      // 2C. ReLU. Obtain Hidden2 from Hidden1.
      if DisplaySubstage then Writeln('': Stage, '2C. Transform Forward, Obtain Hidden2 by ReLU Forward from Hidden1 and MLP Dropout');

      // Activation: Input Hidden1. Output Hidden2.
      // Equation: Hidden2 = ReLU(Hidden1).
      LaunchReLUForward(Hidden1.dValue, Hidden2.dValue, SeqLen, ModelDimProj);

      // Do MLP dropout.
      if Training then
        LaunchDropout(Hidden2.dValue, SeqLen * ModelDimProj, MLPDropOut, MLPDropoutSeed);

      // 2D. Multiplication/Overwrite. Obtain X6 from Hidden2.
      if DisplaySubstage then Writeln('': Stage, '2D. Transform Forward, Obtain X6 from Hidden2');

      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      CuMatMulNN(CuHandle, Hidden2.dValue, W2.dValue, X6.dValue, SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from X6 and b2.
      if DisplaySubstage then Writeln('': Stage, '2E. Transform Forward, Obtain X6 from X6 and b2');

      // Addition: Input X6, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{D}.
      LaunchAddBiasRows(X6.dValue, b2.dValue, SeqLen, ModelDim);

      // Display X6.Value matrix.
      if VerboseTransform then begin
        cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X6, in transform, after contraction.', X6.Value, B);
      end;

      // 2F. Addition/Merge and residual dropout. Obtain X7 from X4 and X6.
      if DisplaySubstage then Writeln('': Stage, '2F. Transform Forward, Obtain X7 from X4 and X6, and RDropout');

      // Do residual dropout.
      if Training then
        LaunchDropout(X6.dValue, SeqLen * ModelDim, RDropout, RDropoutSeed);

      // Residual merge: Input X4, X6. Output X7.
      // Equation: X7 = X4 + X6. X7 in R^{L · D}. X4 in R^{L · D}. X6 in R^{L x D}.
      CuMatAdd(CuHandle, X4.dValue, X6.dValue, X7.dValue, SeqLen, ModelDim);

      // Display X7.Value matrix.
      if VerboseTransform then begin
        cudaMemcpy(@X7.Value[0, 0], X7.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X7.Value, in transform, after residual added to X6.', X7.Value, B);
      end;

  end;   // End with WModel.
end;     // End RunTransformForward.

end.
}
