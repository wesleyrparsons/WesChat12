unit Infer;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  GPT2Tokenize,
  Math,
  Matrix,
  OutputHead,
  SysUtils,
  TransformForward,
  WesTokenize,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamState: TWAdamWState);

implementation

// Decode tokens for inference.
function DecodeInferenceToken(const TokenID: Integer): UnicodeString;
begin
  if Tokenizer = WesTokenizer then
    Result := Decode(TokenID)
  else
    Result := DecodeGPT2Token(TokenID);
end;

procedure WriteInferenceTokens(const Tokens: TIVector);
var
  i: Integer;
begin
  if Tokenizer = WesTokenizer then begin
    for i := 0 to High(Tokens) do
      Write(UTF8Encode(Decode(Tokens[i])));
  end
  else
    Write(UTF8Encode(DecodeGPT2Tokens(Tokens)));
end;

// Sameple the top probs.
function SampleTopK(const TopTokVector: array of Integer; const TopProbVector: array of Single;
    out TopTok: Integer; out TopKSampleProb: Single): Integer;
var
  i, LastValid: Integer;
  Total, R, Accum: Double;
begin
  TopTok := EOS;
  TopKSampleProb := 0.0;
  Result := EOS;

  if (Length(TopTokVector) = 0) or (Length(TopProbVector) = 0) then Exit;

  if Length(TopTokVector) <> Length(TopProbVector) then begin
    Writeln('SampleTopK: token and probability vector lengths differ.');
    Exit;
  end;

  Total := 0.0;
  LastValid := -1;

  // Sum the valid top-K probabilities.
  for i := 0 to High(TopProbVector) do begin
    if (TopTokVector[i] < 0) or (TopProbVector[i] <= 0.0) then Continue;

    Total := Total + TopProbVector[i];
    LastValid := i;
  end;

  // No valid candidate.
  if (Total <= 0.0) or (LastValid < 0) then Exit;

  R := Random * Total;
  Accum := 0.0;

  // Select according to the probabilities within the top-K set.
  for i := 0 to High(TopProbVector) do begin
    if (TopTokVector[i] < 0) or (TopProbVector[i] <= 0.0) then Continue;

    Accum := Accum + TopProbVector[i];

    if R <= Accum then begin
      TopTok := TopTokVector[i];

      // Normalized probability within the top-K sampling pool.
      TopKSampleProb := TopProbVector[i] / Total;

      Result := TopTok;
      Exit;
    end;
  end;

  // Floating-point fallback.
  TopTok := TopTokVector[LastValid];
  TopKSampleProb := TopProbVector[LastValid] / Total;
  Result := TopTok;
end;

// Build padded tokenizedquery.
procedure BuildInferenceInputTokens(var InputTokens: TIDimVector; const QueryTokenized: TIVector; const SeqLen: Integer; out LastPos: Integer);
var
  i, CopyLen, SrcStart: Integer;
begin
  // Fill everything with PAD.
  for i := 0 to SeqLen - 1 do
    InputTokens[i] := PAD;

  if Length(QueryTokenized) >= SeqLen then begin
    // Use most recent SeqLen tokens.
    SrcStart := Length(QueryTokenized) - SeqLen;
    CopyLen := SeqLen;
    LastPos := SeqLen - 1;
  end
  else begin
    SrcStart := 0;
    CopyLen := Length(QueryTokenized);
    LastPos := CopyLen - 1;
  end;

  for i := 0 to CopyLen - 1 do
    InputTokens[i] := QueryTokenized[SrcStart + i];
end;

// Infer a single token.
procedure InferOneToken(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const QueryTokenized: TIVector; var QueryToken: Integer; var QueryProb: Single);
const
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).
  KSample = 5;
var
  j, Blk, LastPos, BestTok, TopTok: Integer;
  TopTokVector: array[0..KSample - 1] of Integer;
  TopProbVector: array[0..KSample - 1] of Single;
  ModelProb, TopKSampleProb: Single;

  procedure TopKProbs;
  var
    t, p, j: Integer;
  begin
    // Initialize.
    for t := 0 to KSample - 1 do begin
      TopTokVector[t] := -1;
      TopProbVector[t] := -1.0;
    end;

    // Find top K.
    for j := 0 to nVocab - 1 do with WModelState do begin
      for p := 0 to KSample - 1 do
        if Probs[LastPos, j] > TopProbVector[p] then begin

          // Shift down.
          for t := KSample - 1 downto p + 1 do begin
            TopTokVector[t] := TopTokVector[t - 1];
            TopProbVector[t] := TopProbVector[t - 1];
          end;

          TopTokVector[p] := j;
          TopProbVector[p] := Probs[LastPos, j];
          Break;
        end;
    end;

    // Optionally display the top picks.
    if VerboseInfer then begin
      Write('Top probability candidates: ');
      for j := 0 to KSample - 1 do begin
        Write(TopProbVector[j]: 9: 7, ' ', TopTokVector[j], ' ');

        if TopTokVector[j] >= 0 then
          Write(UTF8Encode(DecodeInferenceToken(TopTokVector[j])))
        else
          Write('BADTOKEN');

        {if Tokenizer = WesTokenizer then
          Write(Decode(TopTokVector[j]))
        else
          if Assigned(Vocab) and (TopTokVector[j] >= 0) and (TopTokVector[j] < Vocab.Count) then
            Write(DisplayToken(UTF8Decode(Vocab[TopTokVector[j]])))
        else
          Write('BADTOKEN');}
        Write('  ');
      end;

      Writeln;
    end;
  end;

begin
  // Check for valid query.
  if Length(QueryTokenized) = 0 then begin
    QueryToken := EOS;
    QueryProb := 0.0;
    Exit;
  end;

  if VerboseTransform then with WModelParams do begin
    cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
    VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
  end;

  with WModelState do begin
    // Build the input vector.
    BuildInferenceInputTokens(InputTokens, QueryTokenized, SeqLen, LastPos);

    if VerboseInfer then begin
      Write('Step ', LastPos, '. Length(QueryTokenized) = ', Length(QueryTokenized), '.');
      Write('Last 20 InputTokens: ');
      for j := Max(0, LastPos - 19) to LastPos do
        Write(InputTokens[j], ' ');
      Writeln;
    end;

    for j := 0 to LastPos do
      if (InputTokens[j] < 0) or (InputTokens[j] >= nVocab) then
        Writeln('BAD TOKEN at ', j, ': ', InputTokens[j], '. nVocab = ', nVocab, '.');

    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    // Build X only for block 0.
    LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

    // Scale only block 0 input.
    // Optional transformer-style embedding scaling by sqrt(d_model).
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    // Forward pass through stacked transformer blocks.
    for Blk := 0 to nBlock - 1 do begin
      if VerboseTransform then begin
        cudaMemcpy(@StateBlock[Blk].X.Value[0, 0], StateBlock[Blk].X.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X.Value before transform.', StateBlock[Blk].X.Value, B);
      end;

      RunTransformForward(WModelParams, WModelState, Blk);

      // Feed this block's output into the next block's input.
      if Blk < nBlock - 1 then
        // CopyXTensor(WModelState.StateBlock[Blk].X7, WModelState.StateBlock[Blk + 1].X);
        cudaMemcpy(WModelState.StateBlock[Blk + 1].X.dValue, WModelState.StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    // Compute logit and probits.
    RunOutputForward(WModelParams, WModelState);
    cudaMemcpy(@Probs[LastPos, 0], dProbs + LastPos * DimVocab, nVocab * SizeOf(Single), cudaMemcpyDeviceToHost);
    if DebugCudaChecks then
      CheckCudaError('Copy final probability row for inference.');

    // Set special tokens to zero probability.
    if Tokenizer = WesTokenizer then begin
      Probs[LastPos, BOS] := 0.0;
      Probs[LastPos, PAD] := 0.0;
      Probs[LastPos, UNK] := 0.0;
    end
    else begin
      // Suppress only GPT-2-specific custom special IDs that really exist
      // in this model's vocabulary.
      if (GPT2BOS >= 0) and (GPT2BOS < nVocab) then
        Probs[LastPos, GPT2BOS] := 0.0;

      if (GPT2PAD >= 0) and (GPT2PAD < nVocab) then
        Probs[LastPos, GPT2PAD] := 0.0;

      if (GPT2UNK >= 0) and (GPT2UNK < nVocab) then
        Probs[LastPos, GPT2UNK] := 0.0;
    end;
{    Probs[LastPos, BOS] := 0.0;
    Probs[LastPos, PAD] := 0.0;
    Probs[LastPos, UNK] := 0.0;
    // Don't zero EOS, need to stop.}

    // Check that probs are changing.
    if VerboseInfer then begin
      Write('After softmax, probs = ');
      for j := 0 to 9 do
        Write(Probs[LastPos, j]: 7: 7, '  ');
      Writeln;
    end;

    // ArgMax: Pick the largest probs, and save them. Not using this.
    {Writeln('Final Inference Stage, pick largest token probability.');
    BestProb := Probs[LastPos, 0];
    BestTok := 0;
    for j := 1 to nVocab - 1 do
      if Probs[LastPos, j] > BestProb then begin
        BestProb := Probs[LastPos, j];
        BestTok := j;
      end;}

    // Discourage immediate repetition before finding the top K.
    if Length(QueryTokenized) > 0 then
      Probs[LastPos, QueryTokenized[High(QueryTokenized)]] := Probs[LastPos, QueryTokenized[High(QueryTokenized)]] * 0.25;

    // Find the adjusted top-K candidates.
    TopKProbs;

    // Sample one token from the top-K candidates.
    BestTok := SampleTopK(TopTokVector, TopProbVector, TopTok, TopKSampleProb);

    // Raw model probability after special-token suppression and repetition penalty.
    ModelProb := Probs[LastPos, BestTok];

    // Return the selected token and its adjusted model probability.
    QueryToken := BestTok;
    QueryProb := ModelProb;

  end;
end;

// Run inference forward without additional training.
// QueryInput is the query from the user, like Corpus.
// QueryString is the string input from the user.
// QueryOutput is the output tokens from the model.
// QueryTokenized is the tokenization of QueryInput, like TokenizedCorpus.
// QueryToken is the single next token produced by the infer proc.
{procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState);
const
  MaxNewTokens = 500;                  // Limit on new tokens produced.
var
  i, Step, QueryToken: Integer;
  OldNVocab: Integer;
  OldTraining, OldVerboseTransform, OldSaveTokenizationFiles, OwnsCuda: Boolean;
  QueryTokenized, WorkTokens, QueryOutput: TIVector;
  QueryProb: Single;
  QueryInput: TBVector;
  QueryString: string;
begin
  OldTraining := Training;             // Save status of training.
  OldVerboseTransform := VerboseTransform;
  OldSaveTokenizationFiles := SaveTokenizationFiles;
  OldNVocab := nVocab;

  Training := False;                   // Disable transformer dropout during inference.
  VerboseTransform := False;           // Select verbosity during inference.
  SaveTokenizationFiles := False;      // Don't save files when go to Tokenize.

  if Tokenizer = WesTokenizer then
  if nVocab <> Length(SymbolTable) then
    raise Exception.CreateFmt('Inference vocabulary mismatch: model nVocab=%d, symbol table length=%d.', [nVocab, Length(SymbolTable)]);
  //nVocab := nSymbols;                  // Same variable.

  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting inference...');
    Exit;
  end;

  // Initialize transformer state.
  InitializeTransformerState(WModelState);
  OwnsCuda := not CudaAllocated;

  try
    if OwnsCuda then
      StartCuda(WModelParams, WModelState);

    // Send params (if new model) and inverse freq to device.
    if OwnsCuda or ParamsNeedCopyToDevice then
      CopyParamsToDevice(WModelParams);

    CopyInvFreqToDevice(WModelState);}

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamState: TWAdamState);
const
  MaxNewTokens = 500;
var
  i, Step, QueryToken: Integer;
  OldNVocab: Integer;
  OldTraining, OldVerboseTransform, OldSaveTokenizationFiles: Boolean;
  OwnsCuda, StartedCudaHere: Boolean;
  QueryTokenized, WorkTokens, QueryOutput: TIVector;
  QueryProb: Single;
  QueryInput: TBVector;
  QueryString: string;
begin
  OldTraining := Training;
  OldVerboseTransform := VerboseTransform;
  OldSaveTokenizationFiles := SaveTokenizationFiles;
  OldNVocab := nVocab;

  OwnsCuda := False;
  StartedCudaHere := False;

  try
    Training := False;
    VerboseTransform := False;
    SaveTokenizationFiles := False;

    if Tokenizer = WesTokenizer then
      if nVocab <> Length(SymbolTable) then
        raise Exception.CreateFmt('Inference vocabulary mismatch: model nVocab=%d, symbol table length=%d.', [nVocab, Length(SymbolTable)]);

    if nVocab > DimVocab then begin
      Writeln('nVocab > DimVocab. Aborting inference...');
      Exit;
    end;

    InitializeTransformerState(WModelState);

    OwnsCuda := not CudaAllocated;

    if OwnsCuda then begin
      StartCuda(WModelParams, WModelState, WAdamState);
      StartedCudaHere := True;
    end;

    if OwnsCuda or ParamsNeedCopyToDevice then begin
      CopyParamsToDevice(WModelParams);

      // Keep this only if CopyParamsToDevice does not already do it.
      ParamsNeedCopyToDevice := False;
    end;

    CopyInvFreqToDevice(WModelState);

    // Query loop.
    while True do begin

        // Get a query from user.
      Write('Enter query, blank to return: ');
      Readln(QueryString);

      if UpCase(QueryString) = 'V' then begin
        VerboseInfer := not (VerboseInfer);
        Writeln('Verbose infer = ', VerboseInfer);
      end;

      if (QueryString = EmptyStr) or (UpCase(QueryString) = 'X') or (UpCase(QueryString) = 'EXIT') then begin
        Writeln('Leaving inference.');
        Break;
      end;

      if (QueryString = '?') or (UpCase(QueryString) = 'H') or (UpCase(QueryString) = 'HELP') then begin
        Write(DateTimeToStr(Now), '  X = Exit program. V = toggle Verbose mode. I = program Information. ');
      end;

      if (QueryString = 'I') or (UpCase(QueryString) = 'INFO') then begin
        ReportProgramInfo;
        Writeln('nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' ModelDim = ', ModelDim, ' Projection = ', Proj,
          ' Epoch = ', MaxEpochs, ' Blocks = ', nBlock, ' Heads = ', nHead);
      end;

      Writeln('Query string: ', QueryString);

      SetLength(QueryInput, Length(QueryString));
      for i := 0 to Length(QueryString) - 1 do
        QueryInput[i] := Ord(QueryString[i + 1]);

      if VerboseInfer then begin
        Write(Length(QueryInput), ' ', 'Query String as Bytes: ');
        for i := 0 to Length(QueryInput) - 1 do
           Write(QueryInput[i], ' ');
        Writeln;
      end;

      if Tokenizer = WesTokenizer then
        TokenizeWesBytes(QueryInput, QueryTokenized)
      else
        RunGPT2Tokenize(QueryString, QueryTokenized); // This is wrong.

      if (Length(QueryTokenized) > 0) and (QueryTokenized[High(QueryTokenized)] = EOS) then
        SetLength(QueryTokenized, Length(QueryTokenized) - 1);

      if Length(QueryTokenized) = 0 then begin
        Writeln('No input tokens remain after removing EOS.');
        Exit;
      end;

      if VerboseInfer and (Tokenizer = WesTokenizer) then
        TCFull(QueryTokenized);

      SetLength(QueryOutput, 0);
      WorkTokens := Copy(QueryTokenized);

      for Step := 1 to MaxNewTokens do begin
        // Run Infer to get one additional token.
        InferOneToken(WModelParams, WModelState, WorkTokens, QueryToken, QueryProb);

        // Add the newly generated token to the output.
        SetLength(QueryOutput, Length(QueryOutput) + 1);
        QueryOutput[High(QueryOutput)] := QueryToken;

        // Stop after recording EOS.
        if QueryToken = EOS then Break;

        // EOS does not need to be placed back into the next input.
        SetLength(WorkTokens, Length(WorkTokens) + 1);
        WorkTokens[High(WorkTokens)] := QueryToken;

        if VerboseInfer then begin
          Writeln('Single query token number = ', QueryToken, '. Probability = ', QueryProb: 6: 6, '. Decoded = <<', UTF8Encode(DecodeInferenceToken(QueryToken)), '>>.');
        end;
        {if VerboseInfer then begin
          if Tokenizer = WesTokenizer then
            // WesTokenizer.
            Writeln('Single Query Token Number = ', QueryToken, ' Probability = ', QueryProb: 6: 6,
              ' Decoded = <<', Decode(QueryToken), '>>');
            Writeln('Inference. nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' ModelDim = ', ModelDim, ' Projection = ', Proj,
              '  Epoch = ', MaxEpochs, ' Blocks = ', nBlock, ' Heads = ', nHead);
          // else
            // GPT2Tokenizer.  Need to add this.
            // Writeln('Single Query Token Number = ', QueryToken, ' Probability = ', WModelState.Probs[SeqLen - 1, QueryToken]: 6: 6,
            // ' Decoded = ', DisplayToken(UTF8Decode(Vocab[QueryToken])));

            {Write('Cumulative Decode: ');
            for i := 0 to High(QueryOutput) do
              // WesTokenizer.
              // Write(SymbolTable[QueryOutput[i]]);
              Write(Decode(QueryOutput[i]));
            Writeln;}}

            Write('WorkTokens: <<');
            WriteInferenceTokens(WorkTokens);
            Writeln('>>');
            {Write('WorkTokens: <<');
            for i := 0 to High(WorkTokens) do
              Write(Decode(WorkTokens[i]));
            Writeln('>>');}
          Pause;
      end;

      Write('Query decoded token output: ');
      WriteInferenceTokens(QueryOutput);
      Writeln;
      {Write('Query token output: ');
      for i := 0 to High(QueryOutput) do
        Write(QueryOutput[i], ' ');
      Writeln;}

      if Tokenizer = GPT2Tokenizer then begin
        // GPT2.
        Write('Query decoded token output: ');
        for i := 0 to High(QueryOutput) do
          if Assigned(Vocab) and (QueryOutput[i] >= 0) and (QueryOutput[i] < Vocab.Count) then
            Write(DisplayToken(UTF8Decode(Vocab[QueryOutput[i]])))
          else
            Writeln('BAD TOKEN at ', i, ': ', QueryOutput[i], '. nVocab = ', nVocab, '.');
        Writeln;
      end
      else begin
        // WesChat.
        Write('Query decoded token output: ');
        DetokenizeToDisplay(QueryOutput, F);
        Writeln;
      end;
      Writeln;
    end;
  finally
    try
      if StartedCudaHere then
        EndCuda(WModelParams, WModelState, WAdamState);
    finally
      Training := OldTraining;
      VerboseTransform := OldVerboseTransform;
      SaveTokenizationFiles := OldSaveTokenizationFiles;
      nVocab := OldNVocab;
    end;
  end;
  {finally
    Training := OldTraining;
    VerboseTransform := OldVerboseTransform;
    SaveTokenizationFiles := OldSaveTokenizationFiles;
    nVocab := OldNVocab;

    if OwnsCuda then
      EndCuda(WModelParams, WModelState);
  end;}
end;

end.

