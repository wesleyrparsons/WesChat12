unit Transform;

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

procedure RunTransform(var WModel: WModelType);

implementation

// Run the transformer.
procedure RunTransform(var WModel: WModelType);
var
  h, i, j, HeadOffset: Integer;
begin
  // Display entry to transform.
  writeln('Entering Transformer/FFN/Head Output');

  // Zero gradients.
  ZeroGradients(WModel);

  // Display X.Value matrix.
  VTPDisplayX('Display X.Value, beginning, in transform, before any action.', X.Value, G);

  with WModel do begin
  // BLOCK 0.

  // 1. FORWARD STAGE: ATTENTION.

    // 1A. Layer-Norm. Obtain X1 from X.

    // Layer Norm: Input X. Output X1.
    // Obtain input X from Tokenizer for Transformer stage.
    // Purpose: Normalization.
    // Equation: X1 = LayerNorm(X). X, X1 in R^{L × D}. Gamma1, Beta1 in R^{D}.
    LayerNormForward(X.Value, X1.Value, SeqLen, Gamma1.Value, Beta1.Value, LNXhat1, LNInvStd1);

    // Display X1.Value matrix.
    VTPDisplayX('Display X1.Value, beginning, after layer-norming.', X1.Value, B);

    // 1B. Split. Implicit split into X1 and accumulate into X4.

    // 1C. Multiplication/Overwrite. Obtain Q, K, V from X1.

    // Full Size Multiplication/Overwrite: Input X1, Wq. Output Q.
    // Equation: Q = X1 · Wq. Q in R^{L x D}. X1 in R^{L · D}. Wq in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
     MatMulNN(@X1.Value[0, 0], @Wq.Value[0, 0], @Q.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // Display Q.Value matrix.
    VTPDisplayX('Display Q in transform.', Q.Value, G);

    // Full Size Multiplication/Overwrite: Input X1, Wk. Output K.
    // Equation: K = X1 · Wk. K in R^{L x D}. X1 in R^{L · D}. Wk in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
      MatMulNN(@X1.Value[0, 0], @Wk.Value[0, 0], @K.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // Display K.Value matrix.
    VTPDisplayX('Display K, end, in transform.', K.Value, E);

    // Full Size Multiplication/Overwrite: Input X1, Wv. Output V.
    // Equation: V = X1 · Wv. V in R^{L x D}. X1 in R^{L · D}. Wv in R^{D x D}. M=SeqLen N=ModelDim K=ModelDim.
      MatMulNN(@X1.Value[0, 0], @Wv.Value[0, 0], @V.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // 1D. RoPE.
      ApplyRoPE(Q.Value, InvFreq, SeqLen, ModelDim);
      ApplyRoPE(K.Value, InvFreq, SeqLen, ModelDim);

    // 1E. Multiplication. Obtain Scores1.

    // Multihead Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=HeadDim

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // Q_h is Q[*, headOffset .. headOffset+H-1]
      // K_h is K[*, headOffset .. headOffset+H-1]
      // Multiply Q_h (L x H) by K_h^T (H x L)
      cblas_sgemm(RowMajor, NoTrans, Trans, SeqLen, SeqLen, HeadDim, 1.0, @Q.Value[0, HeadOffset],
        ModelDim, @K.Value[0, HeadOffset], ModelDim, 0.0, @ScoresHead1[h].Value[0, 0], SeqLen);
      {OR THIS WAY:
      MatMulNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0], SeqLen, SeqLen, HeadDim);}
    end;

    // Display ScoresHead[0].Value matrix.
    VTPDisplayX('Display ScoresHead1[0] before standardizing.', ScoresHead1[0].Value, B);

    // 1F. Mask & Softmax & Dropout. Obtain Scores2.

    // Standardization: ScoresHead1 = Sqrt(1 / HeadDim). Now done below, in V, as a scale.

    // Masking: Input ScoresHead1. Output ScoresHead1.
    // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
    for h := 0 to nHead - 1 do
      ApplyAutoRegressiveMask(ScoresHead1[h].Value, SeqLen);

    // Softmax: Input ScoresHead1. Output ScoresHead2.
    // Equation: ScoresHead2 = Softmax(ScoresHead1). ScoresHead in R^{L x L}.
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        SoftmaxForward(ScoresHead1[h].Value[i], ScoresHead2[h].Value[i]);

    // Display Scores1Head2[1].Value matrix.
    VTPDisplayX('Display ScoresHead2[1] after softmax, in transform, before any action.', ScoresHead2[1].Value, G);

    // Do attention dropout.
    // Equation: ScoresHead2 = Dropout(ScoresHead2). ScoresHead in R^{L x L}.
    if Training then
      for h := 0 to nHead - 1 do
        for i := 0 to SeqLen - 1 do
          for j := 0 to SeqLen - 1 do
            if Random < ADropout then
              ScoresHead2[h].Value[i, j] := 0.0
            else
              ScoresHead2[h].Value[i, j] := ScoresHead2[h].Value[i, j] / (1.0 - ADropOut);

    // 1G. Multiplication/Overwrite. Obtain X2Head from ScoresHead2.

    // Scoring: Input ScoresHead2, VHead. Output: X2Head.
    // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      cblas_sgemm(RowMajor, NoTrans, NoTrans, SeqLen, HeadDim, SeqLen, 1.0,
        @ScoresHead2[h].Value[0,0], SeqLen,
        @V.Value[0, HeadOffset], ModelDim,
        InvSqrtHeadDim,
        @X2.Value[0, HeadOffset], ModelDim   // write directly into final L×D output
      );
    end;

    // Display X2.Value matrix.
    VTPDisplayX('Display X2, after Softmax, and concatenation.', X2.Value, B);

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.

    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    MatMulNN(@X2.Value[0, 0], @W0.Value[0, 0], @X3.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // Display X3.Value matrix.
    VTPDisplayX('Display X3, in transform.', X2.Value, B);

    // 1I. Merge. Obtain X4 from X1 and X3.

    // Merge Addition: Input X1, X3. Output X4.
    // Equation: X4 = X1 + X3. X4 in R^{L · D}. X1 in R^{L · D}. X2 in R^(L x D}.
    MatAdd(X1.Value, X3.Value, X4.Value, SeqLen, ModelDim);

    // Display X4.Value matrix.
    VTPDisplayX('Display X4.Value, in transform, after residual added to X3.', X4.Value, G);

    // 1J. Layer-Norm. Obtain X5 from X4.

    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    LayerNormForward(X4.Value, X5.Value, SeqLen, Gamma2.Value, Beta2.Value, LNXhat2, LNInvStd2);

    // Display X5.Value matrix.
    VTPDisplayX('Display X5.Value, in transform, before FFN.', X5.Value, G);

      // 2. STAGE FORWARD FFN.

      // 2A. Multiplication/Overwrite. Obtain Hidden1 from X5 and W1.

      // Expansion: Input X5, W1. Output Hidden1.
      // Equation: Hidden1 = X5 · W1. Hidden1 in R^{L x DB}. X5 in R^{L x D}. W1 in R^{D x DB}.
      MatMulNN(@X5.Value[0, 0], @W1.Value[0, 0], @Hidden1.Value[0, 0], SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden1 from Hidden1 and b1.

      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 * b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      // AddMatVec(@Hidden1.Value, b1.Value, SeqLen, ModelDimProj);
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDimProj,  1.0,  @b1.Value[0], 1,  @Hidden1.Value[i,0], 1);

      // Display Hidden1.Value matrix.
      VTPDisplayX('Display Hidden1.Value, in transform,  after adding b1, and before ReLU.', Hidden1.Value, G);

      // 2C. ReLU. Obtain Hidden2 from Hidden1.

      // Activation: Input Hidden1. Output Hidden2.
      // Equation: Hidden2 = ReLU(Hidden1).
      ReLUMaskForward(Hidden1.Value, Hidden2.Value);

      // Do attention dropout.
      if Training then
        for i := 0 to SeqLen - 1 do
          for j := 0 to ModelDimProj - 1 do
            if Random < RDropout then
              Hidden2.Value[i, j] := 0.0
            else
              Hidden2.Value[i, j] := Hidden2.Value[i, j] / (1.0 - RDropOut);

      // 2D. Multiplication/Overwrite. Obtain X6 from Hidden2.

      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      MatMulNN(@Hidden2.Value[0, 0], @W2.Value[0, 0], @X6.Value[0, 0], SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from Hidden2 and b2.

      // Addition: Input Hidden2, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{L x D}.
      // AddMatVec(@X6.Value, @b2.Value, SeqLen, ModelDim);
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDim,  1.0,  @b2.Value[0], 1,  @X6.Value[i,0], 1);

      // Display X6.Value matrix.
      VTPDisplayX('Display X6, in transform, after contraction.', X6.Value, B);

      // 2F. Addition/Merge. Obtain X7 from X5 and X6.

      // Backprop Merge Addition: Input Residual X6, X5. Output X7.
      // Equation: X7 = X5 + X6. X7 in R^{L · D}. X5 in R^{L · D}. X6 in R^{L x D}.
      MatAdd(X5.Value, X6.Value, X7.Value, SeqLen, ModelDim);

      // Display X7.Value matrix.
      VTPDisplayX('Display X7.Value, in transform, after residual added to X6.', X7.Value, B);

      // 3. FORWARD HEAD OUTPUT STAGE.

        // 3A. Multiplication/Overwrite. Obtain Logits from X7 and Vocab.

        // Multiplication: Input X7, Vocab. Output Logits.
      { // Equation: Logits = X7 · WVocab. Logits in R^{L x nVocab}. X in R^{L x D}.  WVocab in R^{D x nVocab}.
           MatMulNN(@X7.Value[0, 0], @WVocab.Value[0, 0], @Logits[0, 0], SeqLen, nVocab, ModelDim);}

        { Use Embeddings {nVocab x D}, instead of WVocab, doing weight-tying }
        // Equation: Logits = X7 · Embeddings-T. Logits in R^{L x nVocab}. X in R^{L x D}.  Embeddingsin R^{nVocab x D}.
           MatMulNT(@X7.Value[0, 0], @Embeddings.Value[0, 0], @Logits[0, 0], SeqLen, nVocab, ModelDim);
                             // chnage last 2 params above??
        // Display Logits matrix.
        VTPDisplayX('Display Logits, in transform, before softmax.', Logits, B);

        { // Display WVocab.Value matrix.
        VTPDisplayX('Display WVocab.Value in transform, before computing Logit.', WVocab.Value, B); }

        // Display Embeddings.Value matrix.
        // VTPDisplayX('Display Embeddings.Value in transform, before computing Logit.', Embeddings.Value, B);
           //Need display for Embeddings.

        // 3B. Softmax. Obtain Logits from Logits.

        // Softmax: Input Logit. Output Logit.
        // Equation: Logit = Softmax(Logit).
        for i := 0 to SeqLen - 1 do
          SoftmaxForward(Logits[i], Logits[i]);

        // Display Logits matrix.
        VTPDisplayX('Display Logits, in transform, after softmax.', Logits, B);

        // 3C. Cross-Entropy Loss. Obtain TopGradient from Logits.
        // Gradient: Input Logits. Output TopGradient.
        // Equation: TopGradient in R^{L x nVocab}. Logits in R^{L x nVocab}.
        GradientFromProbabilities;

   //fis this.     // Display TopGradient matrix.
        VTPDisplayX('Display TopGradient, in transform, after Logit calculation.', TopGradient, B);

      // BACK PROPAGATION. FEED BACKWARD NETWORK.

      // 2F. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.

      { // Equation: X7.Grad = TopGradient · WVocabᵀ.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. WVocabᵀ in R^{nVocab x D}.
      writeln('Stage 2F');
      cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, nVocab,  1.0,   // 112 = Transposed.
        @TopGradient[0, 0], DimVocab,  @WVocab.Value[0, 0], DimVocab, 0.0,  @X7.Grad[0, 0], ModelDim); }

      // Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings in R^{D x nVocab}.
      writeln('Stage 2F');
      cblas_sgemm(101, 111, 112,  SeqLen, ModelDim, nVocab,  1.0,
        @TopGradient[0, 0], DimVocab,  @Embeddings.Value[0, 0], DimVocab, 0.0,  @X7.Grad[0, 0], ModelDim);

      // Backprop TopGradient modifies/overwrites WVocab: Input X7ᵀ, TopGradient. Output WVocab.Grad.
      // Equation: WVocab.Grad = X7ᵀ · TopGradient. WVocab.Grad in R^{D x nVocab}. X7ᵀ in R^(D x L). TopGradient in R^{L x nVocab}.
      { cblas_sgemm(101, 112, 111,  ModelDim, nVocab, SeqLen,  1.0,  @X7.Value[0, 0], ModelDim,
        @TopGradient[0,0], DimVocab,  1.0,  @.Grad[0,0], DimVocab); }

      // Backprop TopGradient modifies/overwrites Embeddings-T: Input X7ᵀ, TopGradient. Output Embeddings-T.Grad.
      // Equation: Embeddings-T.Grad = X7ᵀ · TopGradient. Embeddings-T.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
      cblas_sgemm(101, 112, 111,  ModelDim, nVocab, SeqLen,  1.0,  @X7.Value[0, 0], ModelDim,
        @TopGradient[0,0], DimVocab,  1.0,  @Embeddings.Grad[0,0], DimVocab);

      // Backprop Split X7 Grad into X5 and X6: Input X5.Grad, X7.Grad. Output dX.Grad.
      // Equation: X5.Grad = X5.Grad + X7.Grad. All in R^{L x D}.
      GradSplit(X7.Grad, X5.Grad, X6.Grad, SeqLen, ModelDim);

      // Display X7.Grad matrix.
      VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', X7.Grad, G);

      // 2E. Backprop Addition/Accumulation. Obtain b2 from X6.

      Writeln('Stage 2E');
      // Backprop X6 Grad creates b2 Grad. Input X6.Grad. Output b2.Grad.
      // Equation: b2.Grad = sum of X6.Grad. b2.Grad is R^{L x D}. X6.Grad in R^{L x D}.
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDim,  1.0,  @X6.Grad[i, 0], 1,  @b2.Grad[0], 1);

      // 2D. Backprop Multiplication/Overwrite. Obtain W2 from Hidden2 and X6.

      Writeln('Stage 2D');
      // Backprop X6 Grad creates W2 Grad: Input Hidden2ᵀ.Value, X6.Grad. Output W2.Grad.
      // Equation: W2.Grad = Hidden2ᵀ.Value · X6.Grad. W2.Grad is R^{DB x D}. Hidden2ᵀ.Value is R^{DB x L}. X6.Grad in R^{L x D}.
      MatMulTN(@Hidden2.Value, @X6.Grad, @W2.Grad, ModelDimProj, ModelDim, SeqLen);

      // Backprop X6 Grad creates Hidden2 Grad: Input
      // Equation: Hidden2.Grad = X6.Grad * W2ᵀ.Value. X6.Grad in R^{L x D}. W2ᵀ.Value is R^{D x DB}. Hidden2.Grad is R^{L x DB}.
      MatMulNT(@X6.Grad, @W2.Value, @Hidden2.Grad, SeqLen, ModelDimProj, ModelDim);

      // 2C. Backprop ReLU. Obtain Hidden1 from Hidden2.

      Writeln('Stage 2C');
      // Backprop BackReLU activation on Hidden: Input Hidden2.Grad. Output Hidden1.Grad.
      // Equation: Hidden1.Grad = ReLUMaskBackward(Hidden2.Grad). Hidden1.Grad is R^{L x DB}. Hidden2.Value is R^{L x DB}.
      for i := 0 to SeqLen - 1 do
        for j := 0 to ModelDimProj - 1 do
          if Hidden1.Value[i, j] > 0.0 then
            Hidden1.Grad[i, j] := Hidden2.Grad[i, j]
          else
            Hidden1.Grad[i, j] := 0.0;

      // 2B. Backprop Addition/Accumulate. Obtain b1 from Hidden1.

      Writeln('Stage 2B');
      // Backprop Hidden Grad creates b1 Grad: Input Hidden1.Grad. Output b1.Grad.
      // Equation: b1.Grad = sum of Hidden1.Grad. b1.Grad is R^{L x DB}. Hidden1.Grad in R^{L x DB}.
      for i := 0 to SeqLen - 1 do
        cblas_saxpy(ModelDimProj,  1.0,  @Hidden1.Grad[i, 0], 1,  @b1.Grad[0], 1);

      // 2A. Backprop Multiplication/Overwrite. Obtain W1 from X5ᵀ and Hidden1.

      // Obtain X5 from Hidden1 and W1.
      Writeln('Stage 2A');
      // Backprop Hidden1 Grad creates W1 Grad. Input: X5ᵀ.Value, Hidden1.Grad. Output: W1.Grad.
      // Equation: W1.Grad = X5ᵀ.Value · Hidden1.Grad. W1.Grad is R^{D x DB}. X5ᵀ.Value is R^{L x D}. Hidden1.Grad is R^{D x DB).
      MatMulTN(@X5.Value, @Hidden1.Grad, @W1.Grad, SeqLen, ModelDimProj, ModelDim);

      // Backprop Hidden1 Grad accumulates into X5 Grad. Input: Hidden1.Grad, W1ᵀ.Value. Output: X5.Grad.
      // Equation: X5.Grad = Hidden1.Grad · W1ᵀ.Value. Hidden1.Grad is R^{D x DB). W1ᵀ.Value is R^{DB x D}. X5.Grad is R^{L x D}.
      MatMulAccNT(@Hidden1.Grad, @W1.Value, @X5.Grad, SeqLen, ModelDim, ModelDimProj);

    // 1. BACKPROP STAGE TRANSFORMER.

    // 1J. Backprop Layer-Norm. Obtain X5 from X4.

    Writeln('Stage 1J');
    // Backprop Layer-Norm: Input X5, dX5. Output X4.Grad, Gamma2.Grad, Beta2.Grad.
    // Equation: X4.Grad, Gamma2.Grad, Beta2.Grad = LayerNorm(X5, X5.Grad, Gamma2, Beta2). X4.Grad, X5.Grad in R^{L x D}. Gamma2.Grad, Beta2.Grad in R^{D}.
    LayerNormBackward(X5.Grad, X4.Grad, Gamma2.Grad, Beta2.Grad, SeqLen, Gamma2.Value, LNXhat2, LNInvStd2);

    // Display X4.Grad matrix.
    VTPDisplayX('Display X4.Grad, in transform, after stage 1J, layer-norm.', X4.Grad, G);

    // 1I. Backprop Split. Input: X1.Grad. Output: X3.Grad. Output X4.Grad,

    Writeln('Stage 1I');
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

    Writeln('Stage 1G');
    // Equations: Scores2.Grad = X2.Grad · Vᵀ.Value. Scores2.Grad is R^{L x L}. X2.Grad is R^{L x D}. Vᵀ.Value is R^{D x L}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      cblas_sgemm(RowMajor, NoTrans, Trans, SeqLen, SeqLen, HeadDim, 1.0,
        @X2.Grad[0, HeadOffset], ModelDim,  @V.Value[0, HeadOffset], ModelDim, 0.0, @ScoresHead2[h].Grad[0, 0], SeqLen);
    end;

    // Backprop Create VHead Grad from X2Head Grad: Input ScoresHead2ᵀ.Value, X2Head.Grad. Output: VHead.Grad.
    // Equations: VHead.Grad = ScoresHead2ᵀ.Value · X2Head.Grad. VHead.Grad is R^{L x D}. ScoresHead2ᵀ.Value is R^{L x L}. X2Head.Grad is R^{L x D}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      cblas_sgemm(
        RowMajor,
        Trans,       // Scores_h^T
        NoTrans,     // dX_h
        SeqLen,                // M = L
        HeadDim,                // N = H
        SeqLen,                // K = L
        1.0,
        @ScoresHead2[h].Value[0, 0], SeqLen,       // lda = L
        @X2.Grad[0, HeadOffset], ModelDim,    // ldb = D
        InvSqrtHeadDim,
        @V.Grad[0, HeadOffset], ModelDim      // ldc = D
      );
    end;

    // 1F. Backprop Standardize, Mask & Softmax. Obtain ScoresHead1.
    Writeln('Stage 1F');
    // Insure ScoresHead1.Grad is empty.
    for h := 0 to nHead - 1 do
      FillChar(ScoresHead1[h].Grad, SizeOf(ScoresHead1[h].Grad), 0);
    // Backprop Softmax: Input ScoresHead2.Value ScoresHead2.Grad. Output ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = SoftMaxBackwards(ScoresHead2.Value, ScoresHead2.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        SoftmaxBackward(ScoresHead2[h].Value[i], ScoresHead2[h].Grad[i], ScoresHead1[h].Grad[i]);

    // Backprop AutoRegression.
    // Equation: ScoresHead1.Grad = Unmask(ScoresHead1.Grad).
    for h := 0 to nHead - 1 do
      for i := 0 to SeqLen - 1 do
        for j := i + 1 to SeqLen - 1 do
          ScoresHead1[h].Grad[i, j] := 0.0;

    // Backprop standardization. Input: ScoresHead1.Grad. Output: ScoresHead1.Grad.
    // Equation: ScoresHead1.Grad = Sqrt(1 / ModelDim). ScoresHead1.Grad in R^{L x L}.
    // This is now done in the cblas, for V, above.

    // Display ScoresHead.Grad matrix.
    VTPDisplayX('ScoresHead1[0].Grad, transform, before stage 1E, Q and K-transform.', ScoresHead1[0].Grad, G);

    // 1E. Backprop multiplication. Obtain QHead.Grad and KHead.Grad.

    Writeln('Stage 1E');
    // Backprop Multiplication: Input ScoresHead1.Grad, KHead.Value. Output QHead.Grad.
    // Equation: QHead.Grad = ScoresHead1.Grad · KHead.Value. QHead.Grad, ScoresHead1.Grad in R^{L x L}. KHead.Value in R^{L x D}.

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      cblas_sgemm(RowMajor, NoTrans, NoTrans, SeqLen, HeadDim, SeqLen, 1.0,  // Move scalehere" SqrtD?
        @ScoresHead1[h].Grad[0,0], SeqLen, @K.Value[0, HeadOffset], ModelDim, 0.0, @Q.Grad[0, HeadOffset], ModelDim);
    end;

    // Backprop Multiplication: Input ScoresHead1.Gradᵀ, Q.Value. Output K.Grad.
    // Equation: K.Grad = ScoresHead1.Gradᵀ · Q.Value. K.Grad in R^{L x D}. ScoresHead1.Gradᵀ in R^{L · L}. Q.Value in R^{L x D}.
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      cblas_sgemm(
        RowMajor,
        Trans,
        NoTrans,
        SeqLen,
        HeadDim,
        SeqLen,
        1.0,
        @ScoresHead1[h].Grad[0,0], SeqLen,
        @Q.Value[0, HeadOffset], ModelDim,
        0.0,
        @K.Grad[0, HeadOffset], ModelDim
      );
    end;

    // 1D. RoPE. Nil.

    // 1C. Backprop multiplication/overwrite. Obtain W_.Grad and X1_q.Grad for Q, K, and V.

    Writeln('Stage 1D');
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
    MatMulTN(@X1.Value, @K.Grad, @Wk.Grad[h], ModelDim, ModelDim, SeqLen);

    // Backprop Create X1kk Grad from K Grad. Input K.Grad, Wkᵀ.Value. Output X1k.Grad.
    // Equation: X1k.Grad = K.Grad · Wkᵀ.Value. X1k.Grad in R^{L x D}. K.Grad in R^{L x D}. Wkᵀ.Value in R^{D · D}.
    MatMulNT(@K.Grad, @Wk.Value, @X1k.Grad, SeqLen, ModelDim, ModelDim);

    {Wv.Grad = X1ᵀ · V.Grad
     X1v.Grad = V.Grad · Wvᵀ.Value}
    // Backprop Create Wv Grad from V Grad: Input X1ᵀ.Value · V.Grad. Output Wv.Grad.
    // Equation: Wv.Grad = X1ᵀ.Value · V.Grad. Wv.Grad in R^{D x D}. X1ᵀ in R^{D x L}. V.Grad in R^{L x D}.
    MatMulTN(@X1.Value, @V.Value, @Wv.Grad, ModelDim, ModelDim, SeqLen);

    // Backprop Create X1v Grad from V Grad. Input V.Grad, Wvᵀ. Value. Output X1v.Grad.
    // Equation: X1v.Grad = V.Grad times Wvᵀ.Value. X1v.Grad = V.Grad · WVᵀ.Value. V.Grad in R^{L x D}. Wvᵀ.Value in R^{D · D}.
    MatMulNT(@V.Grad, @Wv.Value, @X1v.Grad, SeqLen, ModelDim, ModelDim);

    // 1B. Backprop Merge: Obtain X1 Grad as sum of Grads. Input X1q.Grad, X1k.Grad, and X1v.Grad. Output X1.Grad.
    Writeln('Stage 1B');
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
    Writeln('Stage 1A');
    // Equation: X.Grad, Gamma1.Grad, Beta1.Grad = LayerNorm(X1.Value, X1.Grad, Gamma1.Value, Beta1.Value). X.Grad, X1.Grad in R^{L x D}. Gamma1.Grad, Beta1.Grad in R^{D}.
    LayerNormBackward(X1.Grad, X.Grad, Gamma1.Grad, Beta1.Grad, SeqLen, Gamma1.Value, LNXhat1, LNInvStd1);

    // Display X.Grad matrix.
    VTPDisplayX('Display X.Grad, in transform, at end.', X.Grad, G);

    // Modify weights and biases.
    Optimization(WModel);

    // Place X1 in X for next block.
    CopyXMatrix(X1.Value, X.Value, SeqLen, ModelDim);
    If VerboseTransform then begin
      Writeln('End of tranformer block.');
      Pause;
    end;
  end;   // End with WModel.
end;     // End RunTransform.

end.

