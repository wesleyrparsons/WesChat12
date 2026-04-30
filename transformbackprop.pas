unit TransformBackprop;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Matrix,
  Util;

const
  InvSqrtHeadDim: Single = 1 / Sqrt(HeadDim);         // Used in softmax.
  RowMajor = 101;       // Row Major.
  NoTrans  = 111;       // No transposition.
  Trans    = 112;       // Transposition.

procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);

implementation

// Run the transformer.
procedure RunTransformBackprop(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);
var
  h, i, j, k, HeadOffset, BestTok: Integer;
  BestProb: Single;

begin
  // Display entry to transform.
  writeln('Entering Bacxkprop Transformer');

  // Display X.Value matrix.
  VTPDisplayX('Display X.Value in transform, before any action.', WModelState.StateBlock[0].X.Value, G);

  with WModelParams.ParamBlock[Blk] do with WModelState.StateBlock[Blk] do begin

      // BACK PROPAGATION. FEED BACKWARD NETWORK.

      // 2E. Backprop Addition/Accumulation. Obtain b2 from X6.
      Writeln('            Transform Backprop Stage 2E');

      // Backprop X6 Grad creates b2 Grad. Input X6.Grad. Output b2.Grad.
      // Equation: b2.Grad = sum of X6.Grad. b2.Grad is R^{L x D}. X6.Grad in R^{L x D}.
      for i := 0 to SeqLen - 1 do
        AddScaled(ModelDim, 1.0, @X6.Grad[i,0], @b2.Grad[0]);

      // 2D. Backprop Multiplication/Overwrite. Obtain W2 from Hidden2 and X6.
      Writeln('            Transform Backprop Stage 2D');

      // Backprop X6 Grad creates W2 Grad: Input Hidden2ᵀ.Value, X6.Grad. Output W2.Grad.
      // Equation: W2.Grad = Hidden2ᵀ.Value · X6.Grad. W2.Grad is R^{DB x D}. Hidden2ᵀ.Value is R^{DB x L}. X6.Grad in R^{L x D}.
      MatMulTN(@Hidden2.Value, @X6.Grad, @W2.Grad, ModelDimProj, ModelDim, SeqLen);

      // Backprop X6 Grad creates Hidden2 Grad: Input
      // Equation: Hidden2.Grad = X6.Grad * W2ᵀ.Value. X6.Grad in R^{L x D}. W2ᵀ.Value is R^{D x DB}. Hidden2.Grad is R^{L x DB}.
      MatMulNT(@X6.Grad, @W2.Value, @Hidden2.Grad, SeqLen, ModelDimProj, ModelDim);

      // 2C. Backprop ReLU. Obtain Hidden1 from Hidden2.
      Writeln('            Transform Backprop Stage 2C');

      // Backprop BackReLU activation on Hidden: Input Hidden2.Grad. Output Hidden1.Grad.
      // Equation: Hidden1.Grad = ReLUMaskBackward(Hidden2.Grad). Hidden1.Grad is R^{L x DB}. Hidden2.Value is R^{L x DB}.
      for i := 0 to SeqLen - 1 do
        for j := 0 to ModelDimProj - 1 do
          if Hidden1.Value[i, j] > 0.0 then
            Hidden1.Grad[i, j] := Hidden2.Grad[i, j]
          else
            Hidden1.Grad[i, j] := 0.0;

      // 2B. Backprop Addition/Accumulate. Obtain b1 from Hidden1.
      Writeln('            Transform Backprop Stage 2B');

      // Backprop Hidden Grad creates b1 Grad: Input Hidden1.Grad. Output b1.Grad.
      // Equation: b1.Grad = sum of Hidden1.Grad. b1.Grad is R^{L x DB}. Hidden1.Grad in R^{L x DB}.
      for i := 0 to SeqLen - 1 do
        AddScaled(ModelDimProj, 1.0, @Hidden1.Grad[i,0], @b1.Grad[0]);

      // 2A. Backprop Multiplication/Overwrite. Obtain W1 from X5ᵀ and Hidden1.
      Writeln('            Transform Backprop Stage 2A');

      // Obtain X5 from Hidden1 and W1.
      // Backprop Hidden1 Grad creates W1 Grad. Input: X5ᵀ.Value, Hidden1.Grad. Output: W1.Grad.
      // Equation: W1.Grad = X5ᵀ.Value · Hidden1.Grad. W1.Grad is R^{D x DB}. X5ᵀ.Value is R^{L x D}. Hidden1.Grad is R^{D x DB).
      MatMulTN(@X5.Value, @Hidden1.Grad, @W1.Grad, SeqLen, ModelDimProj, ModelDim);

      // Backprop Hidden1 Grad accumulates into X5 Grad. Input: Hidden1.Grad, W1ᵀ.Value. Output: X5.Grad.
      // Equation: X5.Grad = Hidden1.Grad · W1ᵀ.Value. Hidden1.Grad is R^{D x DB). W1ᵀ.Value is R^{DB x D}. X5.Grad is R^{L x D}.
      MatMulAccNT(@Hidden1.Grad, @W1.Value, @X5.Grad, SeqLen, ModelDim, ModelDimProj);

    // 1. BACKPROP STAGE TRANSFORMER.

    // 1J. Backprop Layer-Norm. Obtain X5 from X4.
    Writeln('          Transform Backprop Stage 1J');

    // Backprop Layer-Norm: Input X5, dX5. Output X4.Grad, Gamma2.Grad, Beta2.Grad.
    // Equation: X4.Grad, Gamma2.Grad, Beta2.Grad = LayerNorm(X5, X5.Grad, Gamma2, Beta2). X4.Grad, X5.Grad in R^{L x D}. Gamma2.Grad, Beta2.Grad in R^{D}.
    LayerNormBackward(X5.Grad, X4.Grad, Gamma2.Grad, Beta2.Grad, SeqLen, Gamma2.Value, LNXhat2, LNInvStd2);

    // Display X4.Grad matrix.
    VTPDisplayX('Display X4.Grad, in transform, after stage 1J, layer-norm.', X4.Grad, G);

    // 1I. Backprop Split. Input: X1.Grad. Output: X3.Grad. Output X4.Grad,
    Writeln('          Transform Bacprop Stage 1I');

    // Equation: X3.Grad, X4.Grad = X1.Grad. All in R^{L x D}.
    GradSplit(X4.Grad, X1.Grad, X3.Grad, SeqLen, ModelDim);

    // Guide: To find the change for the weights: dW0 = X6ᵀ ·  dX7.
    //        To find the error for the input: dX6 = dX7  · W0ᵀ.
    // Guide: To find the change for the multiplication: dScores = dX2 · Vᵀ.
    //        To find the error for the input: dV = Sᵀ · dX2.

    // 1H. Backprop Mutiplication/Overwrite. Obtain W0 Grad from X3 Grad: Input: X2ᵀ.Value, X3.Grad. Output: W0.Grad.

    // Equations: W0.Grad = X2ᵀ.Value · X3.Grad. W0.Grad is R^{L x D}. X3.Grad is R^{L x D}.
    MatMulTN(@X2.Value, @X3.Grad, @W0.Grad, ModelDim, SeqLen, ModelDim);

    // Backprop Create X2 Grad from X3 Grad: Input: X3.Grad, W0ᵀ.Value. Output: X2.Grad.
    // Equations: X2.Grad = X3.Grad · W0ᵀ. W0.Grad is R^{L x D}. X2.Grad, X3.Grad is R^{L x D}. W0ᵀ.Value is R^{D x L}.
    MatMulNT(@X3.Grad, @W0.Value, @X2.Grad, SeqLen, ModelDim, ModelDim);

    // Display X3.Grad matrix.
    VTPDisplayX('Display X3.Grad, in transform, before stage 1G.', X3.Grad, G);

    // 1G. Backprop Multiplication/Overwrite. Obtain Scores2.Grad from X2.Grad: Input X2.Grad, Vᵀ.Value. Output: Scores2.Grad.
    Writeln('          Transform Backprop Stage 1G');

    // Equations: Scores2.Grad = X2.Grad · Vᵀ.Value. Scores2.Grad is R^{L x L}. X2.Grad is R^{L x D}. Vᵀ.Value is R^{D x L}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      MatMulFullTN(@X2.Grad[0, HeadOffset], @V.Value[0, HeadOffset], @ScoresHead2[h].Grad[0,0], SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen);
    end;

    // Backprop Create VHead Grad from X2Head Grad: Input ScoresHead2ᵀ.Value, X2Head.Grad. Output: VHead.Grad.
    // Equations: VHead.Grad = ScoresHead2ᵀ.Value · X2Head.Grad. VHead.Grad is R^{L x D}. ScoresHead2ᵀ.Value is R^{L x L}. X2Head.Grad is R^{L x D}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      MatMulFullNT(@ScoresHead2[h].Value[0,0], @X2.Grad[0, HeadOffset], @V.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // 1F. Backprop Standardize, Mask & Softmax. Obtain ScoresHead1.
    Writeln('          Transform Backprop Stage 1F');

    // Insure ScoresHead1.Grad is empty.
    for h := 0 to nHead - 1 do
      FillChar(ScoresHead1[h].Grad, SizeOf(ScoresHead1[h].Grad), 0);

    // Backprop Softmax: Input ScoresHead2.Value ScoresHead2.Grad. Output ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = SoftMaxBackwards(ScoresHead2.Value, ScoresHead2.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        SoftmaxBackward(ScoresHead2[h].Value[i], ScoresHead2[h].Grad[i], ScoresHead1[h].Grad[i]);

    // Scaling after Softmax.
    for h := 0 to nHead - 1 do
      Scale(SeqLen * SeqLen, InvSqrtHeadDim, @ScoresHead1[h].Grad[0,0]);
    //cblas_sscal(SeqLen * SeqLen, InvSqrtHeadDim, @ScoresHead1[h].Grad[0,0], 1);

    // Backprop AutoRegression.
    // Equation: ScoresHead1.Grad = Unmask(ScoresHead1.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        for j := i + 1 to SeqLen - 1 do
          ScoresHead1[h].Grad[i, j] := 0.0;

    // Backprop standardization. Input: ScoresHead1.Grad. Output: ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = Sqrt(1 / ModelDim). ScoresHead1.Grad in R^{L x L}. Done above.

    // Display ScoresHead.Grad matrix.
    VTPDisplayX('ScoresHead1[0].Grad, transform, before stage 1E, Q and K-transform.', ScoresHead1[0].Grad, G);

    // 1E. Backprop multiplication. Obtain QHead.Grad and KHead.Grad.
    Writeln('          Transform Backprop Stage 1E');

    // Backprop Multiplication: Input ScoresHead1.Grad, KHead.Value. Output QHead.Grad.
    // Equation: QHead.Grad = ScoresHead1.Grad · KHead.Value. QHead.Grad, ScoresHead1.Grad in R^{L x L}. KHead.Value in R^{L x D}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      MatMulFullNN(@ScoresHead1[h].Grad[0,0], @K.Value[0, HeadOffset], @Q.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Backprop Multiplication: Input ScoresHead1.Gradᵀ, Q.Value. Output K.Grad.
    // Equation: K.Grad = ScoresHead1.Gradᵀ · Q.Value. K.Grad in R^{L x D}. ScoresHead1.Gradᵀ in R^{L · L}. Q.Value in R^{L x D}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      MatMulFullNT(@ScoresHead1[h].Grad[0,0], @Q.Value[0, HeadOffset], @K.Grad[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // 1D. RoPE. Nil.
    Writeln('          Transform Backprop Stage 1D Nil');

    // 1C. Backprop multiplication/overwrite. Obtain W_.Grad and X1_q.Grad for Q, K, and V.
    Writeln('          Transform Backprop Stage 1C');

    // Obtain X1q, X1k, X1v, from X1.
    {Wq.Grad = X1ᵀ.Value · Q.Grad
     X1q.Grad = Q.Grad · Wqᵀ.Value}
    // Backprop Create Wq Grad from Q Grad: Input X1ᵀ.Value · Q.Grad. Output Wq.Grad.
    // Equation: Wq.Grad = X1ᵀ · Q.Grad. Wq.Grad in R^{D x D}. X1ᵀ in R^{D x L}. Q.Grad in R^{L x D}.
    MatMulTN(@X1.Value, @Q.Grad, @Wq.Grad, ModelDim, ModelDim, SeqLen);

    // Backprop Create X1q from Q Grad: Input Q.Grad, Wqᵀ.Value. Output X1q.Grad.
    // Equation: X1q.Grad = Q.Grad · Wqᵀ. X1q.Grad in R^{L x D}. Q.Grad in R^{L x D}. Wqᵀ.Value in R^{D · D}.
    MatMulNT(@Q.Grad, @Wq.Value, @X1q.Grad, SeqLen, ModelDim, ModelDim);

    {Wk.Grad = X1ᵀ.Value · K.Grad
     X1k.Grad = K.Grad · Wkᵀ.Value}
    // Backprop Create Wk Grad from K Grad: Input X1ᵀ.Value · K.Grad. Output Wk.Grad.
    // Equation:  Wk.Grad = X1ᵀ.Value · K.Grad. Wk.Grad in R^{D x D}. X1ᵀ.Value in R^{D x L}. K.Grad in R^{L x D}.
    MatMulTN(@X1.Value, @K.Grad, @Wk.Grad, ModelDim, ModelDim, SeqLen);

    // Backprop Create X1kk Grad from K Grad. Input K.Grad, Wkᵀ.Value. Output X1k.Grad.
    // Equation: X1k.Grad = K.Grad · Wkᵀ.Value. X1k.Grad in R^{L x D}. K.Grad in R^{L x D}. Wkᵀ.Value in R^{D · D}.
    MatMulNT(@K.Grad, @Wk.Value, @X1k.Grad, SeqLen, ModelDim, ModelDim);

    {Wv.Grad = X1ᵀ · V.Grad
     X1v.Grad = V.Grad · Wvᵀ.Value}
    // Backprop Create Wv Grad from V Grad: Input X1ᵀ.Value · V.Grad. Output Wv.Grad.
    // Equation: Wv.Grad = X1ᵀ.Value · V.Grad. Wv.Grad in R^{D x D}. X1ᵀ in R^{D x L}. V.Grad in R^{L x D}.
    MatMulTN(@X1.Value, @V.Grad, @Wv.Grad, ModelDim, ModelDim, SeqLen);

    // Backprop Create X1v Grad from V Grad. Input V.Grad, Wvᵀ. Value. Output X1v.Grad.
    // Equation: X1v.Grad = V.Grad times Wvᵀ.Value. X1v.Grad = V.Grad · WVᵀ.Value. V.Grad in R^{L x D}. Wvᵀ.Value in R^{D · D}.
    MatMulNT(@V.Grad, @Wv.Value, @X1v.Grad, SeqLen, ModelDim, ModelDim);

    // 1B. Backprop Merge: Obtain X1 Grad as sum of Grads. Input X1q.Grad, X1k.Grad, and X1v.Grad. Output X1.Grad.
    Writeln('          Transform Backprop Stage 1B');

    // Equation:  X1.Grad = X1q.Grad + X1k.Grad + X1v.Grad. All in R^{L x D}.
    for i := 0 to SeqLen - 1 do
      for j := 0 to ModelDim - 1 do
        X1.Grad[i, j] := X1q.Grad[i, j] + X1k.Grad[i, j] + X1v.Grad[i, j];

    // Backprop Accumulate: Input X1.Grad, X4.Grad. Output X1.Grad.
    // Equation: X1.Grad = X1.Grad + X4.Grad. All R^{L x D}.
    AccumulateGrad(X4.Grad, X1.Grad, SeqLen, ModelDim);

    // Display X3.Grad matrix.
    VTPDisplayX('Display X1.Grad, in transform, after concatenation.', X1.Grad, G);

    // 1A. Backprop Layer-Norm: Input X1.Value, X1.Grad. Output X.Grad, Gamma1.Grad, Beta1.Grad.
    Writeln('          Transform Backprop Stage 1A');

    // Equation: X.Grad, Gamma1.Grad, Beta1.Grad = LayerNorm(X1.Value, X1.Grad, Gamma1.Value, Beta1.Value). X.Grad, X1.Grad in R^{L x D}. Gamma1.Grad, Beta1.Grad in R^{D}.
    LayerNormBackward(X1.Grad, X.Grad, Gamma1.Grad, Beta1.Grad, SeqLen, Gamma1.Value, LNXhat1, LNInvStd1);

    // Display X.Grad matrix.
    VTPDisplayX('Display X.Grad, in transform, at end.', X.Grad, G);

  end;   // End with WModel.
end;     // End RunTransform.

end.

