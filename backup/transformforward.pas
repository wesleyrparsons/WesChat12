unit TransformForward;

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

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);

implementation

procedure RunTransformForward(var WModelParams: TWModelParams; var WModelState: TWModelState; var QueryOutput: TIVector; const Blk: Integer);
// Run the transformer forward.
var
  h, i, j, HeadOffset: Integer;
begin
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
    LayerNormForward(X.Value, X1.Value, SeqLen, Gamma1.Value, Beta1.Value, LNXhat1, LNInvStd1);

    // Display X1.Value matrix.
    VTPDisplayX('Display X1.Value after layer-norming.', X1.Value, B);

    // 1B. Split. Implicit split into X1 and accumulate into X4.
    Writeln('          Transform Forward Stage 1B (Implicit)');

    // 1C. Multiplication/Overwrite. Obtain Q, K, V from X1.
    Writeln('          Transform Forward Stage 1C');

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
    Writeln('          Transform Forward Stage 1E');

    // Multihead Multiplication/Overwrite: Input Q, Kᵀ. Output: Scores1.
    // That is, the Queries * Tansposed(Keys) are the attention scores.
    // Equation: Scores1 = Q · Kᵀ. Scores1 in R^{L · L}. Q in R^{L x D}. Kᵀ in R^{D x L}. M=SeqLen N=SeqLen K=HeadDim

    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;

      // Q_h is Q[*, headOffset .. headOffset+H-1]
      // K_h is K[*, headOffset .. headOffset+H-1]
      // Multiply Q_h (L x H) by K_h^T (H x L), and scale by InvSqrtHeadDim.
      MatMulFullScaledNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0],
        SeqLen, SeqLen, HeadDim, ModelDim, ModelDim, SeqLen, InvSqrtHeadDim, 0.0);
      {OR THIS WAY:
      cblas_sgemm(RowMajor, NoTrans, Trans, SeqLen, SeqLen, HeadDim, InvSqrtHeadDim, @Q.Value[0, HeadOffset],
        ModelDim, @K.Value[0, HeadOffset], ModelDim, 0.0, @ScoresHead1[h].Value[0, 0], SeqLen);
      OR THIS WAY MatMulNT(@Q.Value[0, HeadOffset], @K.Value[0, HeadOffset], @ScoresHead1[h].Value[0, 0], SeqLen, SeqLen, HeadDim);
      And also scale by InvSqrtHeadDim}
    end;

    // Display ScoresHead[0].Value matrix.
    VTPDisplayX('Display ScoresHead1[0] before standardizing.', ScoresHead1[0].Value, B);

    // 1F. Mask & Softmax & Dropout. Obtain Scores2.
    Writeln('          Transform Forward Stage 1F');

    // Masking: Input ScoresHead1. Output ScoresHead1.
    // Equation: ScoresHead1 = Mask(ScoresHead1). ScoresHead1 in R^{L x L}.
    for h := 0 to nHead - 1 do
      ApplyAutoRegressiveMask(ScoresHead1[h].Value, SeqLen);

    // Softmax: Input ScoresHead1. Output ScoresHead2.
    // Equation: ScoresHead2 = Softmax(ScoresHead1). ScoresHead in R^{L x L}.
    // Do not use SoftmaxForwardN here.
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
    Writeln('          Transform Forward Stage 1G');

    // Scoring: Input ScoresHead2, VHead. Output: X.
    // Equation: X2 = Scores2 · V. X2 in R^{L · D}. Scores2 in R^{L x L}. V in R^{L x D}. M=SeqLen N=ModelDim K=SeqLen
    for h := 0 to nHead - 1 do begin
      HeadOffset := h * HeadDim;
      MatMulFullNN(@ScoresHead2[h].Value[0,0], @V.Value[0, HeadOffset], @X2.Value[0, HeadOffset], SeqLen, HeadDim, SeqLen, SeqLen, ModelDim, ModelDim);
    end;

    // Display X2.Value matrix.
    VTPDisplayX('Display X2, after Softmax, and concatenation.', X2.Value, B);

    // 1H. Mutiplication/Overwrite. Obtain X3 by weighting X2 by W0.
    Writeln('          Transform Forward Stage 1H');

    // Weighting: Input X2, W0. Output X3.
    // Equation: X3 = X2 · W0. X3 in R^{L · D}. W0 in R^{D x D}. X2 in R^{L x D}.
    MatMulNN(@X2.Value[0, 0], @W0.Value[0, 0], @X3.Value[0, 0], SeqLen, ModelDim, ModelDim);

    // Display X3.Value matrix.
    VTPDisplayX('Display X3, in transform.', X3.Value, B);

    // 1I. Merge. Obtain X4 from X1 and X3.
    Writeln('          Transform Forward Stage 1I');

    // Merge Addition: Input X1, X3. Output X4.
    // Equation: X4 = X1 + X3. X4 in R^{L · D}. X1 in R^{L · D}. X2 in R^(L x D}.
    MatAdd(X1.Value, X3.Value, X4.Value, SeqLen, ModelDim);

    // Display X4.Value matrix.
    VTPDisplayX('Display X4.Value, in transform, after residual added to X3.', X4.Value, G);

    // 1J. Layer-Norm. Obtain X5 from X4.
    Writeln('          Transform Forward Stage 1J');

    // Layer Norm: Input X4. Output X5.
    // Equation: X5 = LayerNorm(X4). X4 in R^{L × D}. X5 in R^{L × D}. Gamma2, Beta2 in R^{D}.
    LayerNormForward(X4.Value, X5.Value, SeqLen, Gamma2.Value, Beta2.Value, LNXhat2, LNInvStd2);

    // Display X5.Value matrix.
    VTPDisplayX('Display X5.Value, in transform, before FFN.', X5.Value, G);

      // 2. STAGE FORWARD FFN.

      // 2A. Multiplication/Overwrite. Obtain Hidden1 from X5 and W1.
      Writeln('            Transform Forward Stage 2A');

      // Expansion: Input X5, W1. Output Hidden1.
      // Equation: Hidden1 = X5 · W1. Hidden1 in R^{L x DB}. X5 in R^{L x D}. W1 in R^{D x DB}.
      MatMulNN(@X5.Value[0, 0], @W1.Value[0, 0], @Hidden1.Value[0, 0], SeqLen, ModelDimProj, ModelDim);

      // 2B. Addition/Accumulate. Obtain Hidden1 from Hidden1 and b1.
      Writeln('            Transform Forward Stage 2B');

      // Addition: Input Hidden1, b1. Output Hidden1.
      // Equation: Hidden1 = Hidden1 * b1. Hidden1 in R^{L x DB}. b1 in R^{DB}.
      // AddMatVec(@Hidden1.Value, b1.Value, SeqLen, ModelDimProj);
      for i := 0 to SeqLen - 1 do
        AddScaled(ModelDimProj, 1.0, @b1.Value[0], @Hidden1.Value[i,0]);

      // Display Hidden1.Value matrix.
      VTPDisplayX('Display Hidden1.Value, in transform,  after adding b1, and before ReLU.', Hidden1.Value, G);

      // 2C. ReLU. Obtain Hidden2 from Hidden1.
      Writeln('            Transform Forward Stage 2C');

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
      Writeln('            Transform Forward Stage 2D');

      // Contraction: Input Hidden2, W2. Output X6.
      // Equation: X6 = Hidden2 · W2. Hidden2 in R^{L x DB}. W2 in R^{DB x D}. X6 in R^{L x D}.
      MatMulNN(@Hidden2.Value[0, 0], @W2.Value[0, 0], @X6.Value[0, 0], SeqLen, ModelDim, ModelDimProj);

      // 2E. Addition/Accumulation. Obtain X6 from Hidden2 and b2.
      Writeln('            Transform Forward Stage 2E');

      // Addition: Input Hidden2, b2. Output X6.
      // Equation: X6 = X6 + b2. X6 in R^{L x D}. b2 in R^{L x D}.
      // AddMatVec(@X6.Value, @b2.Value, SeqLen, ModelDim);
      for i := 0 to SeqLen - 1 do
        AddScaled(ModelDim, 1.0, @b2.Value[0], @X6.Value[i,0]);

      // Display X6.Value matrix.
      VTPDisplayX('Display X6, in transform, after contraction.', X6.Value, B);

      // 2F. Addition/Merge. Obtain X7 from X5 and X6.
      Writeln('            Transform Forward Stage 2F');

      // Backprop Merge Addition: Input Residual X6, X5. Output X7.
      // Equation: X7 = X5 + X6. X7 in R^{L · D}. X5 in R^{L · D}. X6 in R^{L x D}.
      MatAdd(X5.Value, X6.Value, X7.Value, SeqLen, ModelDim);

      // Display X7.Value matrix.
      VTPDisplayX('Display X7.Value, in transform, after residual added to X6.', X7.Value, B);

  end;   // End with WModel.
end;     // End RunTransform.

end.

