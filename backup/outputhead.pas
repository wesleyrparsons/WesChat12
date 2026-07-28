unit OutputHead;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  Matrix,
  SysUtils,
  Util;

procedure RunOutputForward(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure RunOutputBackward(var WModelParams: TWModelParams; var WModelState: TWModelState);

implementation

// Run the outputhead forward.
procedure RunOutputForward(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  Temperature: Single;
  begin
  // 3. FORWARD HEAD OUTPUT STAGE.
  Stage := nBlock * 4 + 2;

  if Training then
    Temperature := TTemperature
  else
    Temperature := ITemperature;

  with WModelParams do with WModelState do begin

    // 3A. Multiplication/Overwrite. Obtain Logits from X7 and Vocab. Note both Logits and Probs are in Probs.
    if DisplayStage then Writeln('' : Stage, 'Stage 3, Block ', nBlock - 1, ',  Model Output');
    if DisplaySubstage then Writeln('': Stage, '3A. Model Head Forward, Compute Vocabulary Logits');

    // Multiplication: Input X7, Vocab. Output Probs (which are Logits).
    // Equation: Probs = X7 · Embeddingsᵀ. Probs in R^{L x nVocab}. X in R^{L x D}.  Embeddings in R^{nVocab x D}.
    CuMatMulFullNT(CuHandle, StateBlock[nBlock - 1].X7.dValue, Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

    // Infer needs the probs. Done in Infer unit.
    // cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);

    // Display Probs (that is, Logits) matrix.
    if VerboseTransform then begin
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display vocabulary logits before softmax.', Probs, B);
    end;

    // Display Embeddings.Value matrix.
    if VerboseTransform then begin
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value in transform, before computing probabilities.', Embeddings.Value, B);
    end;

    // 3B. Softmax. Obtain probabilities from Softmax(Logit).
    if DisplaySubstage then Writeln('': Stage, '3B. Transform Forward, Softmax Probs');

    // Softmax: Input Probs (Logits). Output Probs.
    // Equation: Probs (Logits) = Softmax(Probs).
    LaunchSoftmaxForwardStrided(dProbs, dProbs, SeqLen, nVocab, DimVocab, Temperature);

    // Display Probs matrix.
    if VerboseTransform then begin
      cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Probs, in transform, after softmax.', Probs, B);
    end;
   end;
  if DebugCudaChecks then
    CheckCudaError('OutputHead forward pass.');
end;

// Run the outputhead backward.
procedure RunOutputBackward(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  GradScale: Single;
begin
  Stage := nBlock * 4 + 2;
  with WModelParams do with WModelState do begin
    // 3C. Cross-entropy gradient. Obtain TopGradient from Probs.
    if DisplaySubstage then
      Writeln('': Stage, '3C. LM Head Backward, Obtain TopGradient from Probs');

    // Average the loss gradient across sequence positions.
    GradScale := 1.0 / (SeqLen * TTemperature);

    LaunchCEGradientStrided(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab, DimVocab, GradScale);

    // Display TopGradient matrix.
    if VerboseTransform then begin
      cudaMemcpy(@TopGradient[0, 0], dTopGradient, ProbsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display TopGradient, in transform.', TopGradient, B);
    end;

    // 3D. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
    if DisplaySubstage then Writeln('': Stage, '3D. Transform Backprop, Create X7 from TopGradient');

    with StateBlock[nBlock - 1] do begin
      // Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings.Value in R^{nVocab x D}.
      CuMatMulFullNN(CuHandle, dTopGradient, Embeddings.dValue, X7.dGrad, SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);

      // Backprop TopGradient accumulate the output-head contribution to the tied embedding Embeddingsᵀ: Input X7ᵀ, TopGradient. Output Embeddingsᵀ.Grad.
      // Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
      CuMatMulFullAccTN(CuHandle, dTopGradient, X7.dValue, Embeddings.dGrad, nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);

      // Display X7.Grad matrix.
      if VerboseTransform then begin
        cudaMemcpy(@X7.Grad[0, 0], X7.dGrad, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X7.Grad, after outputheadbackward, after stage 3D.', X7.Grad, G);
      end;
    end;
  end;
  if DebugCudaChecks then
    CheckCudaError('OutputHead backprop pass.');
end;

end.

