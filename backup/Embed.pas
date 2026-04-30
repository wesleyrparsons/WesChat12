unit Embed;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  IOHandler,
  Math,
  Matrix,
  SysUtils,
  TransformForward,
  TransformBackprop,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunEmbed(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector);
procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector; var QueryOutput: TIVector);

implementation

const
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).

// Create the target vector for use in head output.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector;
  const StartIndex, L: Integer);
var
  i: Integer;
begin
  Assert(StartIndex >= 0);
  Assert(StartIndex + L <= Length(TokenizedCorpus));

  for i := 0 to L - 1 do
    Target[i] := TokenizedCorpus[StartIndex + i];
end;

// Create the input matrix.
procedure BuildInputMatrix(var X: TSeqMatrix; const TokenizedCorpus: TIVector;
  var WModelParams: TWModelParams; const Start, L: Integer);
var
  i, j, id: Integer;
begin
  // Bounds check on the corpus window.
  Assert(Start >= 0);
  Assert(Start + L <= Length(TokenizedCorpus));

  for i := 0 to L - 1 do begin
    id := TokenizedCorpus[Start + i];

    // Validate token ID.
    Assert(id >= 0);
    Assert(id < nSymbols);

    // Copy embedding vector.
    for j := 0 to ModelDim - 1 do
      X[i, j] := WModelParams.Embeddings.Value[id, j];
  end;
end;

// Run the training.
procedure RunEmbed(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector);
var
  i, j, k, Blk: Integer;
  Start, EmbedLoop: Integer;
  BestTok: Integer;
  BestProb: Single;
  InvFreq:    TFVector;           // For RoPE.
  Stride: Integer = 64;      // Stride 64 tokens every sequence.

  procedure ReadEmbedIfKeyPressed;
  var
    key: char;
    Success: Boolean;
    ModelFileName: string;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Pause;
        Halt;                // Immediately terminate program.
      end;
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        Pause;
        Blk := nBlock;     // Break out of the loop cleanly.
      end;
      'v', 'V': begin
        VeryVerbose := not VeryVerbose;
        Writeln('Very verbose mode: ', VeryVerbose);
        Pause;
      end;                   // Change verbosity.
      'i', 'I': begin
        Writeln;
        ReportInfo;          // Report program info.
        Pause;
      end;
      't', 'T': begin
        Writeln('Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
          '  Start = ', Start, ' Stride = ', Stride, ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));
        Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of loop. V = toggle Verbose mode. P = Pause.');
        Writeln('  W = WesChat Information. T = Training information. S = Save. Training...');
        Pause;
      end;
      's', 'S': begin
        ChDir(WorkingDir);   // Save model.
        Write('Enter filename: ');
        Readln(ModelFileName);
        SaveModel(ModelFileName, WModelParams, Success);
        ChDir('..');
        if Success then
          Writeln('File ', f, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
      end;

    end;
  end;

begin
  nVocab := nSymbols;    // Need nVocab (second name for variable) for Transform.

  if VeryVerbose then
    Writeln('Start Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
      ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));

  // Seed the weights with random numbers.
  for i := 0 to nSymbols - 1 do             // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      WModelParams.Embeddings.Value[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  Writeln('First quarter of first row of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModelParams.Embeddings.Value[1, k]: 8: 6, ' ');
  Writeln;
  Pause;

  VTPDisplayX('Display Embeddings.Value prior to Transform.', WModelParams.Embeddings.Value, B);

  // Initialize.
  InitializeTransformer(WModelParams, WModelState);
  SetLength(TokenID, Length(TokenizedCorpus));
  TokenID := TokenizedCorpus;

  // Stride loop thru Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(TokenizedCorpus) do begin

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    Writeln('&&& SeqLen loop: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    Writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
    for k := 0 to nBlock - 1 do
      BuildInputMatrix(WModelState.StateBlock[k].X.Value, TokenizedCorpus, WModelParams, Start, SeqLen);

    // Optional transformer-style embedding scaling by sqrt(d_model).
    for i := 0 to SeqLen - 1 do
      for j := 0 to ModelDim - 1 do
        for k := 0 to nBlock - 1 do
          WModelState.StateBlock[k].X.Value[i, j] := WModelState.StateBlock[k].X.Value[i, j] * Scale;

    // Build the target vector, one ahead, for the loss stage.
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start + 1, SeqLen);

    VTPDisplayX('Display X.Value before transform.', WModelState.StateBlock[0].X.Value, G);

    // Zero gradients.
    for k := 0 to nBlock - 1 do
      ZeroGradients(WModelParams, WModelState, k);

    // Forward pass thru transformer.
    for Blk := 0 to nBlock - 1 do begin
      Writeln('     $$$ Forward Block loop: start ', Blk, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransformForward(WModelParams, WModelState, QueryOutput, Blk);

      if Blk < nBlock then
        CopyXTensor(WModelState.StateBlock[Blk].X7, WModelState.StateBlock[Blk + 1].X1);

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;
    end;

    // 3. FORWARD HEAD OUTPUT STAGE.

    with WModelParams do with WModelState do begin
      // 3A. Multiplication/Overwrite. Obtain Probs from X7 and Vocab.
      Writeln('              Transform Gradient Stage 3A');

      // Multiplication: Input X7, Vocab. Output Probs.
      // Equation: Probs = X7 · Embeddingsᵀ. Probs in R^{L x nVocab}. X in R^{L x D}.  Embeddings in R^{nVocab x D}.
      MatMulFullNT(@StateBlock[Blk].X7.Value[0, 0], @Embeddings.Value[0, 0], @Probs[0, 0], SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

       // Display Probs matrix.
      VTPDisplayX('Display Probs, in transform, before softmax.', Probs, B);

      // Display Embeddings.Value matrix.
      VTPDisplayX('Display Embeddings.Value in transform, before computing Logit.', Embeddings.Value, B);

      // 3B. Softmax. Obtain Probs from Probs.
      Writeln('            Transform Forward Stage 3B');

      // Softmax: Input Logit. Output Logit.
      // Equation: Logit = Softmax(Logit).
      // Use SoftmaxForwardN here.
      for i := 0 to SeqLen - 1 do
        SoftmaxForwardN(@Probs[i,0], @Probs[i,0], nVocab);

      // Display Probs matrix.
      VTPDisplayX('Display Probs, in transform, after softmax.', Probs, B);

      // 3C. If QueryForward, and last nBlock, then pick the largest probs, and save them. Move to end of nBlock loop.
      Writeln('            Transform Forward Stage 3C');
      if not Training and (Blk = nBlock - 1) then begin
        SetLength(QueryOutput, SeqLen);
        for i := 0 to SeqLen - 1 do begin
          BestProb := Probs[i, 0];
          BestTok  := 0;
          for j := 1 to nVocab - 1 do
            if Probs[i, j] > BestProb then begin
              BestProb := Probs[i, j];
              BestTok := j;
            end;
          QueryOutput[i] := BestTok;
        end;
        Exit;
      end;

      // 3D. Cross-Entropy Loss. Obtain TopGradient from Probs.
      Writeln('            Transform Forward Stage 3D');
      // Gradient: Input Probs. Output TopGradient. Also option of CalculateGradient from KLDivergence.
      // Equation: TopGradient in R^{L x nVocab}. Probs in R^{L x nVocab}.
      GradientFromCEProbabilities(WModelState);
      //GradientFromKLDivergence(WModelState);

      // Display TopGradient matrix.
      VTPDisplayX('Display TopGradient, in transform, after Logit calculation.', TopGradient, B);
      // 3E. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
      Writeln('              Transform Backprop Stage 3E');

      // Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings.Value in R^{nVocab x D}.
      MatMulFullNN(@TopGradient[0, 0], @Embeddings.Value[0, 0], @WModelState.StateBlock[Blk].X7.Grad[0, 0], SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);
      {cblas_sgemm(101, 111, 111, SeqLen, ModelDim, nVocab, 1.0, @TopGradient[0, 0], DimVocab,
      @Embeddings.Value[0, 0], ModelDim, 0.0, @X7.Grad[0, 0], ModelDim);}

      Writeln('Finished MatMul X7.Grad loop.');

      // Backprop TopGradient modifies/overwrites Embeddingsᵀ: Input X7ᵀ, TopGradient. Output Embeddingsᵀ.Grad.
      // Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
      // Problem here was I had NT rather than TN.
      MatMulFullAccTN(@TopGradient[0,0], @WModelState.StateBlock[Blk].X7.Value[0,0], @Embeddings.Grad[0,0], nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);
      Writeln('Finished Embeddings.Grad GEMM.');

      // Backprop Split X7 Grad into X5 and X6: Input X5.Grad, X7.Grad. Output dX.Grad.
      // Equation: X5.Grad = X5.Grad + X7.Grad. All in R^{L x D}.
      GradSplit(WModelState.StateBlock[Blk].X7.Grad, WModelState.StateBlock[Blk].X5.Grad, WModelState.StateBlock[Blk].X6.Grad, SeqLen, ModelDim);

      // Display X7.Grad matrix.
      VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', WModelState.StateBlock[Blk].X7.Grad, G);

    end; // End gradient stage.

    // Modify weights and biases.
    for k := 0 to nBlock - 1 do
      Optimization(WModelParams, WModelState, k);

    // Backprop pass thru transformer.
    for Blk := nBlock - 1 downto 0 do begin
      Writeln('     $$$ Backpropd Block loop: start ', Blk, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransformBackprop(WModelParams, WModelState, QueryOutput, Blk);

      if Blk > 0 then
        CopyXTensor(WModelState.StateBlock[Blk].X1, WModelState.StateBlock[Blk - 1].X7);

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;
    end;

    Start := Start + Stride;
  end; // End sequence loop.

  Writeln('End of training. Press <CR> to continue.');
  Readln;
end;

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector; var QueryOutput: TIVector);
var
  i, j, k, Blk: Integer;
  Start, EmbedLoop: Integer;
  Stride: Integer = 64;      // Stride 64 tokens every sequence.

begin
  nVocab := nSymbols;    // Need nVocab (second name for variable) for Transform.

  Writeln('First quarter of two rows of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModelParams.Embeddings.Value[1, k]: 8: 6, ' ');
  Writeln;
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModelParams.Embeddings.Value[2, k]: 8: 6, ' ');
  Writeln;
  Pause;

  VTPDisplayX('Display Embeddings.Value prior to Transform.', WModelParams.Embeddings.Value, B);

  // Initialize.
  SetLength(TokenID, Length(TokenizedCorpus));
  TokenID := TokenizedCorpus;

  // Stride loop thru Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(TokenizedCorpus) do begin

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    Writeln('&&& Loop thru Embed: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    Writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Forward pass thru transformer.
    for Blk := 0 to nBlock - 1 do begin
      Writeln('$$$ Starting Block ', Blk, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
      BuildInputMatrix(WModelState.StateBlock[Blk].X.Value, TokenizedCorpus, WModelParams, Start, SeqLen);

      // Optional transformer-style embedding scaling by sqrt(d_model).
      for i := 0 to SeqLen - 1 do
        for j := 0 to ModelDim - 1 do
          WModelState.StateBlock[Blk].X.Value[i, j] := WModelState.StateBlock[Blk].X.Value[i, j] * Scale;

      // Build the target vector, one ahead, for the loss stage.
      BuildTargetVector(TargetTokens, TokenizedCorpus, Start + 1, SeqLen);

      VTPDisplayX('Display X.Value before transform.', WModelState.StateBlock[Blk].X.Value, G);


      RunTransformForward(WModelParams, WModelState, QueryOutput, Blk);
    end;

    Start := Start + Stride;
  end;

  Writeln('End of training. Press <CR> to continue.');
  Readln;
end;

end.

