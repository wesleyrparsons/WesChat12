unit TransformBackprop;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  Display,
  Global,
  Matrix,
  Util;

procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

{ Transformer backward pass }
// Run one transformer block backward.
procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h, HeadOffset: Integer;
begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    { Feed-forward backward pass }

    Stage := Blk * 4 + 4;

    // Display X6 before starting the feed-forward backward pass.
    if VerboseTransform then begin
      cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X6.Value in transform, before any backpropagation.', X6.Value, G);
    end;

    // 2F. Split the final residual gradient: X7 = X4 + X6.
    if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ', Transform Backprop');
    if DisplaySubstage then Writeln('' : Stage, '2F. Transform Backprop, split X7 gradient and reverse residual dropout');
    CuGradSplit(CuHandle, X7.dGrad, X4.dGrad, X6.dGrad, SeqLen, ModelDim);

    // Reverse residual dropout on the X6 path.
    if Training then
      LaunchDropoutBackward(X6.dGrad, SeqLen * ModelDim, RDropout, RDropoutSeed);

    // 2E. Accumulate the b2 gradient from X6.Grad.
    if DisplaySubstage then Writeln('' : Stage, '2E. Transform Backprop, obtain b2 Grad from X6 Grad');
    // Equation: b2.Grad = sum_rows(X6.Grad). b2.Grad is D; X6.Grad is L x D.
    LaunchAddBiasRowsBackward(X6.dGrad, b2.dGrad, SeqLen, ModelDim);

    // 2D. Backpropagate X6 = Hidden2 * W2.
    if DisplaySubstage then Writeln('' : Stage, '2D. Transform Backprop, obtain W2 Grad and Hidden2 Grad from X6 Grad');

    // W2.Grad = Hidden2^T * X6.Grad. Hidden2 is L x DB; X6.Grad is L x D; W2.Grad is DB x D.
    CuMatMulFullTN(CuHandle, Hidden2.dValue, X6.dGrad, W2.dGrad, ModelDimProj, ModelDim, SeqLen, ModelDimProj, ModelDim, ModelDim);

    // Hidden2.Grad = X6.Grad * W2^T. Hidden2.Grad is L x DB.
    CuMatMulNT(CuHandle, X6.dGrad, W2.dValue, Hidden2.dGrad, SeqLen, ModelDimProj, ModelDim);

    // Reverse MLP dropout on Hidden2.
    if Training then
      LaunchDropoutBackward(Hidden2.dGrad, SeqLen * ModelDimProj, MLPDropout, MLPDropoutSeed);

    // 2C. Backpropagate the ReLU activation: Hidden2 -> Hidden1.
    if DisplaySubstage then Writeln('' : Stage, '2C. Transform Backprop, ReLU backward from Hidden2 to Hidden1');
    LaunchReLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);

    // 2B. Accumulate the b1 gradient from Hidden1.Grad.
    if DisplaySubstage then Writeln('' : Stage, '2B. Transform Backprop, obtain b1 Grad from Hidden1 Grad');
    // Equation: b1.Grad = sum_rows(Hidden1.Grad). b1.Grad is DB; Hidden1.Grad is L x DB.
    LaunchAddBiasRowsBackward(Hidden1.dGrad, b1.dGrad, SeqLen, ModelDimProj);

    // 2A. Backpropagate Hidden1 = X5 * W1.
    if DisplaySubstage then Writeln('' : Stage, '2A. Transform Backprop, obtain W1 Grad and X5 Grad from Hidden1 Grad');

    // W1.Grad = X5^T * Hidden1.Grad. X5 is L x D; Hidden1.Grad is L x DB; W1.Grad is D x DB.
    CuMatMulFullTN(CuHandle, X5.dValue, Hidden1.dGrad, W1.dGrad, ModelDim, ModelDimProj, SeqLen, ModelDim, ModelDimProj, ModelDimProj);

    // X5.Grad += Hidden1.Grad * W1^T. X5.Grad is L x D.
    CuMatMulAccNT(CuHandle, Hidden1.dGrad, W1.dValue, X5.dGrad, SeqLen, ModelDim, ModelDimProj);

    { Attention backward pass }

    Stage := Stage - 2;

    // 1J. Backpropagate X5 = LayerNorm(X4).
    if DisplayStage then Writeln('' : Stage, 'Stage  1, Block ', Blk, ', Transform Backprop');
    if DisplaySubstage then Writeln('' : Stage, '1J. Transform Backprop, LayerNorm backward from X5 to X4');

    // LayerNorm backward produces the X4 path gradient plus Gamma2 and Beta2 gradients.
    LaunchLayerNormBackward(X5.dGrad, dX4FromLN2, Gamma2.dValue, dLNXHat2, dLNInvStd2, Gamma2.dGrad, Beta2.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, dX4FromLN2, X4.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X4.Grad[0, 0], X4.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Grad after LayerNorm backward.', X4.Grad, G);
    end;

    // 1I. Split the attention residual gradient: X4 = X + X3.
    if DisplaySubstage then Writeln('' : Stage, '1I. Transform Backprop, split X4 Grad into X Grad and X3 Grad');
    CuGradSplit(CuHandle, X4.dGrad, X.dGrad, X3.dGrad, SeqLen, ModelDim);

    // 1H. Backpropagate X3 = X2 * W0.
    if DisplaySubstage then Writeln('' : Stage, '1H. Transform Backprop, obtain W0 Grad and X2 Grad from X3 Grad');

    // W0.Grad = X2^T * X3.Grad. X2 and X3.Grad are L x D; W0.Grad is D x D.
    CuMatMulFullTN(CuHandle, X2.dValue, X3.dGrad, W0.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X2.Grad = X3.Grad * W0^T. X2.Grad is L x D.
    CuMatMulNT(CuHandle, X3.dGrad, W0.dValue, X2.dGrad, SeqLen, ModelDim, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X3.Grad[0, 0], X3.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3.Grad before attention backpropagation.', X3.Grad, G);
    end;

    // 1G-E. Backpropagate the per-head attention path.
    if DisplaySubstage then Writeln('' : Stage, '1G-E. Transform Backprop, attention value path, dropout, softmax, mask, Q and K');

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // 1G. ScoresHead2.Grad = X2h.Grad * Vh^T. ScoresHead2.Grad is L x L.
      CuMatMulFullNT(CuHandle, PSingle(X2.dGrad) + HeadOffset, PSingle(V.dValue) + HeadOffset,
        ScoresHead2[h].dGrad, SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen);

      // Vh.Grad = ScoresHead2^T * X2h.Grad. Vh.Grad is L x H.
      CuMatMulFullTN(CuHandle, ScoresHead2[h].dValue, PSingle(X2.dGrad) + HeadOffset,
        PSingle(V.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // 1F-c. Reverse attention dropout.
      if Training then
        LaunchDropoutBackward(ScoresHead2[h].dGrad, SeqLen * SeqLen, ADropout, ADropoutSeed + h);

      // 1F-b. Backpropagate softmax. ScoresHead1.dValue contains the saved pre-dropout softmax probabilities.
      LaunchSoftmaxBackward(ScoresHead1[h].dValue, ScoresHead2[h].dGrad, ScoresHead1[h].dGrad, SeqLen, SeqLen);

      // 1E. Forward attention scores were scaled by 1 / sqrt(HeadDim), so scale their gradient likewise.
      CuScale(CuHandle, SeqLen * SeqLen, InvSqrtHeadDim, ScoresHead1[h].dGrad);

      // 1F-a. Zero gradients in positions suppressed by the autoregressive mask.
      LaunchAutoRegressiveMaskBackward(ScoresHead1[h].dGrad, SeqLen);

      // Qh.Grad = ScoresHead1.Grad * Kh. Qh.Grad is L x H.
      CuMatMulFullNN(CuHandle, ScoresHead1[h].dGrad, PSingle(K.dValue) + HeadOffset,
        PSingle(Q.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // Kh.Grad = ScoresHead1.Grad^T * Qh. Kh.Grad is L x H.
      CuMatMulFullTN(CuHandle, ScoresHead1[h].dGrad, PSingle(Q.dValue) + HeadOffset,
        PSingle(K.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead1[0].Grad[0, 0], ScoresHead1[0].dGrad, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display ScoresHead1[0].Grad before Q and K projection backpropagation.', ScoresHead1[0].Grad, G);
    end;

    // 1D. Reverse RoPE on Q.Grad and K.Grad.
    if DisplaySubstage then Writeln('' : Stage, '1D. Transform Backprop, RoPE backward');
    LaunchRoPEBackward(Q.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEBackward(K.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    // 1C. Backpropagate the Q, K, and V projections from X1.
    if DisplaySubstage then Writeln('' : Stage, '1C. Transform Backprop, obtain Wq/Wk/Wv Grads and X1 path Grads');

    // Wq.Grad = X1^T * Q.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, Q.dGrad, Wq.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1q.Grad = Q.Grad * Wq^T.
    CuMatMulNT(CuHandle, Q.dGrad, Wq.dValue, X1q.dGrad, SeqLen, ModelDim, ModelDim);

    // Wk.Grad = X1^T * K.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, K.dGrad, Wk.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1k.Grad = K.Grad * Wk^T.
    CuMatMulNT(CuHandle, K.dGrad, Wk.dValue, X1k.dGrad, SeqLen, ModelDim, ModelDim);

    // Wv.Grad = X1^T * V.Grad.
    CuMatMulFullTN(CuHandle, X1.dValue, V.dGrad, Wv.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // X1v.Grad = V.Grad * Wv^T.
    CuMatMulNT(CuHandle, V.dGrad, Wv.dValue, X1v.dGrad, SeqLen, ModelDim, ModelDim);

    // 1B. Merge the Q, K, and V paths into X1.Grad.
    if DisplaySubstage then Writeln('' : Stage, '1B. Transform Backprop, sum Q, K, and V paths into X1 Grad');

    // X1.dGrad was zeroed at the beginning of this training window.
    CuAccumulateGrad(CuHandle, X1q.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1k.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1v.dGrad, X1.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X1.Grad[0, 0], X1.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Grad after summing the Q, K, and V paths.', X1.Grad, G);
    end;

    // 1A. Backpropagate X1 = LayerNorm(X).
    if DisplaySubstage then begin
      Writeln('' : Stage, '1A. Transform Backprop, LayerNorm backward from X1 to X');
      PauseNNL;
    end;

    // LayerNorm backward produces the X path gradient plus Gamma1 and Beta1 gradients.
    LaunchLayerNormBackward(X1.dGrad, dXFromLN1, Gamma1.dValue, dLNXHat1, dLNInvStd1, Gamma1.dGrad, Beta1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, dXFromLN1, X.dGrad, SeqLen, ModelDim);

    if VerboseTransform then begin
      cudaMemcpy(@X.Grad[0, 0], X.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Grad at the end of transform backpropagation.', X.Grad, G);
    end;

  end;
end;

end.

{unit TransformBackprop;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com }

interface

uses
  Display,
  Global,
  Matrix,
  Util;

procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

implementation

// Run the transformer backprop.
procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h, HeadOffset: Integer;

begin
  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

    Stage := Blk * 4 + 4;
    // Display X6.Value matrix. X6.Value is in cublas.
    if VerboseTransform then begin
      cudaMemcpy(@X6.Value[0, 0], X6.dValue, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X6.Value in transform, before any action.', X6.Value, G);
    end;

    // BACK PROPAGATION. FEED BACKWARD NETWORK.

      // 2E. Backprop Split and Addition/Accumulation. Backprop through final residual. Obtain b2 from X6.
      if DisplayStage then Writeln('' : Stage, 'Stage 1, Block ', Blk, ', Transform Backprop');
      if DisplaySubstage then Writeln('': Stage, '2E. Transform Backprop, Split, Residual Dropout');

      // Backprop Split X7 Grad into X4 and X6: Input X4.Grad, X7.Grad. Output dX.Grad.
      // Forward: X7 = X4 + X6.
      // Backward:// X4.Grad := X7.Grad; X6.Grad := X7.Grad.
      CuGradSplit(CuHandle, X7.dGrad, X4.dGrad, X6.dGrad, SeqLen, ModelDim);

      // Dropout Backward.
      if Training then
        LaunchDropoutBackward(X6.dGrad, SeqLen * ModelDim, RDropout, RDropoutSeed);

      // Backprop X6 Grad creates b2 Grad. Input X6.Grad. Output b2.Grad.
      // Equation: b2.Grad = sum of X6.Grad. // b2.Grad is R^{D}. X6.Grad in R^{L x D}.
      LaunchAddBiasRowsBackward(X6.dGrad, b2.dGrad, SeqLen, ModelDim);

      // 2D. Backprop Multiplication/Overwrite. Obtain W2 from Hidden2 and X6.
      if DisplaySubstage then Writeln('': Stage, '2D. Transform Backprop, Backprop X6 Grad creates W2 Grad and Backprop X6 Grad creates Hidden2 Grad');

      // Backprop X6 Grad creates W2 Grad: Input Hidden2ᵀ.Value, X6.Grad. Output W2.Grad.
      // Equation: W2.Grad = Hidden2ᵀ.Value · X6.Grad. W2.Grad is R^{DB x D}. Hidden2ᵀ.Value is R^{DB x L}. X6.Grad in R^{L x D}.
      CuMatMulFullTN(CuHandle, Hidden2.dValue, X6.dGrad, W2.dGrad, ModelDimProj, ModelDim, SeqLen, ModelDimProj, ModelDim, ModelDim);

      // Backprop X6 Grad creates Hidden2 Grad: Input
      // Equation: Hidden2.Grad = X6.Grad * W2ᵀ.Value. X6.Grad in R^{L x D}. W2ᵀ.Value is R^{D x DB}. Hidden2.Grad is R^{L x DB}.
      CuMatMulNT(CuHandle, X6.dGrad, W2.dValue, Hidden2.dGrad, SeqLen, ModelDimProj, ModelDim);

      // Dropout Backward.
      if Training then
        LaunchDropoutBackward(Hidden2.dGrad, SeqLen * ModelDimProj, MLPDropout, MLPDropoutSeed);

      // 2C. Backprop ReLU. Obtain Hidden1 from Hidden2.
      if DisplaySubstage then Writeln('': Stage, '2C. Transform Backprop, ReLU and Obtain Hidden 1 from Hidden2');

      // Backprop BackReLU activation on Hidden: Input Hidden2.Grad. Output Hidden1.Grad.
      // Equation: Hidden1.Grad = ReLUMaskBackward(Hidden2.Grad). Hidden1.Grad is R^{L x DB}. Hidden2.Value is R^{L x DB}.
      LaunchReLUBackward(Hidden1.dValue, Hidden2.dGrad, Hidden1.dGrad, SeqLen, ModelDimProj);

      // 2B. Backprop Addition/Accumulate. Obtain b1 from Hidden1.
      if DisplaySubstage then Writeln('': Stage, '2B. Transform Backprop , Obtain b1 from Hidden1');

      // Backprop Hidden Grad creates b1 Grad: Input Hidden1.Grad. Output b1.Grad.
      // Equation: b1.Grad = sum of Hidden1.Grad. b1.Grad is R^{DB}. Hidden1.Grad in R^{L x DB}.
      LaunchAddBiasRowsBackward(Hidden1.dGrad, b1.dGrad, SeqLen, ModelDimProj);

      // 2A. Backprop Multiplication/Overwrite. Obtain W1 from X5ᵀ and Hidden1.
      if DisplaySubstage then Writeln('': Stage, '2A. Transform Backprop, Obtain X5 from Hidden1 and W1');

      // Backprop Hidden1 Grad creates W1 Grad. Input: X5ᵀ.Value, Hidden1.Grad. Output: W1.Grad.
      // Equation: W1.Grad = X5ᵀ.Value · Hidden1.Grad. W1.Grad is R^{D x DB}. X5ᵀ.Value is R^{L x D}. Hidden1.Grad is R^{D x DB).
      CuMatMulFullTN(CuHandle, X5.dValue, Hidden1.dGrad, W1.dGrad, ModelDim, ModelDimProj, SeqLen, ModelDim, ModelDimProj, ModelDimProj);

      // Backprop Hidden1 Grad accumulates into X5 Grad. Input: Hidden1.Grad, W1ᵀ.Value. Output: X5.Grad.
      // Equation: X5.Grad = Hidden1.Grad · W1ᵀ.Value. Hidden1.Grad is R^{D x DB). W1ᵀ.Value is R^{DB x D}. X5.Grad is R^{L x D}.
      CuMatMulAccNT(CuHandle, Hidden1.dGrad, W1.dValue, X5.dGrad, SeqLen, ModelDim, ModelDimProj);

      // 1. BACKPROP STAGE TRANSFORMER.
    Stage := Stage - 2;
    // 1J. Backprop LayerNorm: X5 = LayerNorm(X4, Gamma2, Beta2).
    // Input:  X5.Grad, Gamma2.Value, cached LNXhat2, cached LNInvStd2. Output: X4.Grad, Gamma2.Grad, Beta2.Grad.
    // X5.Grad, X4.Grad, LNXhat2 is R^{L x D}. Gamma2.Value, Gamma2.Grad, Beta2.Grad is R^{D}. LNInvStd2 is R^{L}.
    if DisplayStage then Writeln('' : Stage, 'Stage  2, Block ', Blk, ',  Transform Backprop');
    if DisplaySubstage then Writeln('': Stage, '1J. Transform Backprop Stage 1J, Layer Norm Backward X5');

    // Backprop Layer-Norm: Input X5.Grad, Gamma2.Grad, Beta2.Value. Output X4.Grad, Gamma2.Grad, Beta2.Grad.
    // Equation: X4.Grad, Gamma2.Grad, Beta2.Grad = LayerNorm(X5.Grad, Gamma2.Value, Beta2.Value).
    LaunchLayerNormBackward(X5.dGrad, dX4FromLN2, Gamma2.dValue, dLNXHat2, dLNInvStd2, Gamma2.dGrad, Beta2.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, dX4FromLN2, X4.dGrad, SeqLen, ModelDim);

    // Display X4.Grad matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X4.Grad[0, 0], X4.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X4.Grad, in transform, after stage 1J, layer-norm.', X4.Grad, G);
    end;

    // 1I. Backprop Split. Input: X1.Grad. Output: X3.Grad. Output X4.Grad,
    if DisplaySubstage then Writeln('': Stage, '1I. Transform Backprop Stage 1I, Split X4 = X + X3');

    // Equation: X4 = X + X3, so dX += dX4 and dX3 += dX4. All in R^{L x D}.
    cuGradSplit(cuHandle, X4.dGrad, X.dGrad, X3.dGrad, SeqLen, ModelDim);

    // To find the change for the weights: dW0 = X6ᵀ ·  dX7.
    // To find the error for the input: dX6 = dX7  · W0ᵀ.
    // To find the change for the multiplication: dScores = dX2 · Vᵀ.
    // To find the error for the input: dV = Sᵀ · dX2.

    // 1H. Backprop Mutiplication/Overwrite. Obtain W0 Grad from X3 Grad: Input: X2ᵀ.Value, X3.Grad. Output: W0.Grad.
    if DisplaySubstage then Writeln('': Stage, '1H. Transform Backprop, Obtain W0 Grad from X3 Grad');

    // Equations: W0.Grad = X2ᵀ.Value · X3.Grad. // W0.Grad is R^{D x D}. X3.Grad is R^{L x D}.
    CuMatMulFullTN(CuHandle, X2.dValue, X3.dGrad, W0.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // Backprop Create X2.Grad from X3.Grad: Input: X3.Grad, W0ᵀ.Value. Output: X2.Grad.
    // Equations: X2.Grad = X3.Grad · W0ᵀ. W0.Grad is R^{L x D}. X2.Grad, X3.Grad is R^{L x D}. W0ᵀ.Value is R^{D x L}.
    CuMatMulNT(CuHandle, X3.dGrad, W0.dValue, X2.dGrad, SeqLen, ModelDim, ModelDim);

    // Display X3.Grad matrix. X3.Grad already in cblas.
    if VerboseTransform then begin
      cudaMemcpy(@X3.Grad[0, 0], X3.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X3.Grad, in transform, before stage 1G.', X3.Grad, G);
    end;

    if DisplaySubstage then Writeln('': Stage, '1G-E, Transform Backprop, Obtain Scores1, Autoregressive mask, obtain Scores2, Softmax, ADropout');

    // 1G. Backprop Multiplication/Overwrite. Obtain Scores2.Grad from X2.Grad: Input X2.Grad, Vᵀ.Value. Output: Scores2.Grad.
    // Equations: Scores2.Grad = X2.Grad · Vᵀ.Value. Scores2.Grad is R^{L x L}. X2.Grad is R^{L x D}. Vᵀ.Value is R^{D x L}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      CuMatMulFullNT(CuHandle, PSingle(X2.dGrad) + HeadOffset, PSingle(V.dValue) + HeadOffset,
        ScoresHead2[h].dGrad, SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen);

      // Backprop Create VHead Grad from X2Head Grad: Input ScoresHead2ᵀ.Value, X2Head.Grad. Output: VHead.Grad.
      // Equations: VHead.Grad = ScoresHead2ᵀ.Value · X2Head.Grad. VHead.Grad is R^{L x D}. ScoresHead2ᵀ.Value is R^{L x L}. X2Head.Grad is R^{L x D}.
      CuMatMulFullTN(CuHandle, ScoresHead2[h].dValue, PSingle(X2.dGrad) + HeadOffset,
        PSingle(V.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // Dropout Backward.
      if Training then
        LaunchDropoutBackward(ScoresHead2[h].dGrad, SeqLen * SeqLen, ADropout, ADropoutSeed + h);

      // 1F. Backprop Standardize, Mask & Softmax. Obtain ScoresHead1.

      // Backprop Softmax: Input ScoresHead2.Value ScoresHead2.Grad. Output ScoresHead1.Grad.
      // Equation: ScoresHead1.Grad = SoftMaxBackwards(ScoresHead2.Value, ScoresHead2.Grad).
      // ScoresHead1.dValue contains the pre-dropout softmax output.
      // ScoresHead2.dGrad has already passed through dropout backward.
      LaunchSoftmaxBackward(ScoresHead1[h].dValue, ScoresHead2[h].dGrad, ScoresHead1[h].dGrad, SeqLen, SeqLen);

      // Scaling after Softmax.
      CuScale(CuHandle, SeqLen * SeqLen, InvSqrtHeadDim, ScoresHead1[h].dGrad);

      // Backprop AutoRegression.
      // Equation: ScoresHead1.Grad = Unmask(ScoresHead1.Grad).
      LaunchAutoRegressiveMaskBackward(ScoresHead1[h].dGrad, SeqLen);

      // Backprop standardization. Input: ScoresHead1.Grad. Output: ScoresHead1.Grad.
      // Equation: ScoresHead1.Grad = Sqrt(1 / ModelDim). ScoresHead1.Grad in R^{L x L}. Done above.

      // 1E. Backprop multiplication. Obtain QHead.Grad and KHead.Grad.
      // Backprop Multiplication: Input ScoresHead1.Grad, KHead.Value. Output QHead.Grad.
      // Equation: QHead.Grad = ScoresHead1.Grad · KHead.Value. QHead.Grad, ScoresHead1.Grad in R^{L x L}. KHead.Value in R^{L x HeadDim}.
      CuMatMulFullNN(CuHandle, ScoresHead1[h].dGrad, PSingle(K.dValue) + HeadOffset,
        PSingle(Q.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);

      // Backprop Multiplication: Input ScoresHead1.Gradᵀ, Q.Value. Output K.Grad.
      // Equation: K.Grad = ScoresHead1.Gradᵀ · Q.Value. K.Grad in R^{L x D}. ScoresHead1.Gradᵀ in R^{L · L}. Q.Value in R^{L x D}.
      CuMatMulFullTN(CuHandle, ScoresHead1[h].dGrad, PSingle(Q.dValue) + HeadOffset,
        PSingle(K.dGrad) + HeadOffset, SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end; // h loop.

    // Display ScoresHead.Grad matrix.
    if VerboseTransform then begin
      cudaMemcpy(@ScoresHead1[0].Grad[0, 0], ScoresHead1[0].dGrad, ScoresSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('ScoresHead1[0].Grad, transform, before stage 1E, Q and K-transform.', ScoresHead1[0].Grad, G);
    end;

    // 1D. RoPE Backward.
    if DisplaySubstage then Writeln('': Stage, '1D. Transform Backprop, RoPE');

    LaunchRoPEBackward(Q.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);
    LaunchRoPEBackward(K.dGrad, WModelState.dInvFreq, SeqLen, nHead, HeadDim, ModelDim);

    // 1C. Backprop multiplication/overwrite. Obtain W_.Grad and X1_q.Grad for Q, K, and V.
    if DisplaySubstage then Writeln('': Stage, '1C. Transform Backprop, Obtain W_.Grad and X1_q.Grad for Q, K, and V');

    // Obtain X1q, X1k, X1v, from X1.
    {Wq.Grad = X1ᵀ.Value · Q.Grad
     X1q.Grad = Q.Grad · Wqᵀ.Value}
    // Backprop Create Wq.Grad from Q.Grad: Input X1ᵀ.Value · Q.Grad. Output Wq.Grad.
    // Equation: Wq.Grad = X1ᵀ · Q.Grad. Wq.Grad in R^{D x D}. X1ᵀ in R^{D x L}. Q.Grad in R^{L x D}.
    CuMatMulFullTN(CuHandle, X1.dValue, Q.dGrad, Wq.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // Backprop Create X1q from Q.Grad: Input Q.Grad, Wqᵀ.Value. Output X1q.Grad.
    // Equation: X1q.Grad = Q.Grad · Wqᵀ. X1q.Grad in R^{L x D}. Q.Grad in R^{L x D}. Wqᵀ.Value in R^{D · D}.
    CuMatMulNT(CuHandle, Q.dGrad, Wq.dValue, X1q.dGrad, SeqLen, ModelDim, ModelDim);

    {Wk.Grad = X1ᵀ.Value · K.Grad
     X1k.Grad = K.Grad · Wkᵀ.Value}
    // Backprop Create Wk Grad from K Grad: Input X1ᵀ.Value · K.Grad. Output Wk.Grad.
    // Equation:  Wk.Grad = X1ᵀ.Value · K.Grad. Wk.Grad in R^{D x D}. X1ᵀ.Value in R^{D x L}. K.Grad in R^{L x D}.
    CuMatMulFullTN(CuHandle, X1.dValue, K.dGrad, Wk.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // Backprop Create X1k.Grad from K.Grad. Input K.Grad, Wkᵀ.Value. Output X1k.Grad.
    // Equation: X1k.Grad = K.Grad · Wkᵀ.Value. X1k.Grad in R^{L x D}. K.Grad in R^{L x D}. Wkᵀ.Value in R^{D · D}.
    CuMatMulNT(CuHandle, K.dGrad, Wk.dValue, X1k.dGrad, SeqLen, ModelDim, ModelDim);

    {Wv.Grad = X1ᵀ · V.Grad
     X1v.Grad = V.Grad · Wvᵀ.Value}
    // Backprop Create Wv.Grad from V.Grad: Input X1ᵀ.Value · V.Grad. Output Wv.Grad.
    // Equation: Wv.Grad = X1ᵀ.Value · V.Grad. Wv.Grad in R^{D x D}. X1ᵀ in R^{D x L}. V.Grad in R^{L x D}.
    CuMatMulFullTN(CuHandle, X1.dValue, V.dGrad, Wv.dGrad, ModelDim, ModelDim, SeqLen, ModelDim, ModelDim, ModelDim);

    // Backprop Create X1v.Grad from V.Grad. Input V.Grad, Wvᵀ. Value. Output X1v.Grad.
    // Equation: X1v.Grad = V.Grad times Wvᵀ.Value. X1v.Grad = V.Grad · WVᵀ.Value. V.Grad in R^{L x D}. Wvᵀ.Value in R^{D · D}.
    CuMatMulNT(CuHandle, V.dGrad, Wv.dValue, X1v.dGrad, SeqLen, ModelDim, ModelDim);

    // Equation:  X1.Grad = X1q.Grad + X1k.Grad + X1v.Grad. All in R^{L x D}.
    // 1B. Backprop Merge: X1.Grad = X1q.Grad + X1k.Grad + X1v.Grad.
    if DisplaySubstage then Writeln('': Stage, '1B. Transform Backprop, Obtain X1 Grad as sum of Q, K, and V paths');

    // X1.dGrad was zeroed at the beginning of this training window.
    CuAccumulateGrad(CuHandle, X1q.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1k.dGrad, X1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, X1v.dGrad, X1.dGrad, SeqLen, ModelDim);

    // Display X1.Grad matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X1.Grad[0, 0], X1.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X1.Grad, in transform, after concatenation.', X1.Grad, G);
    end;

    // 1A. Backprop Layer-Norm: Input X1.Value, X1.Grad. Output X.Grad, Gamma1.Grad, Beta1.Grad.
    if DisplaySubstage then begin
      Writeln('': Stage, '1A. Transform Backprop, Layer Norm X1');
      PauseNNL
    end;

    // Equation: X.Grad, Gamma1.Grad, Beta1.Grad = LayerNorm(X1.Value, X1.Grad, Gamma1.Value, Beta1.Value). X.Grad, X1.Grad in R^{L x D}. Gamma1.Grad, Beta1.Grad in R^{D}.
    LaunchLayerNormBackward(X1.dGrad, dXFromLN1, Gamma1.dValue, dLNXHat1, dLNInvStd1, Gamma1.dGrad, Beta1.dGrad, SeqLen, ModelDim);
    CuAccumulateGrad(CuHandle, dXFromLN1, X.dGrad, SeqLen, ModelDim);

    // Display X.Grad matrix.
    if VerboseTransform then begin
      cudaMemcpy(@X.Grad[0, 0], X.dGrad, XSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display X.Grad, in transform, at end.', X.Grad, G);
    end;

  end;   // End with WModel.
end;     // End RunTransform.

end.}

