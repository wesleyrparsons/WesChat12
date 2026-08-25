unit Infer;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  DateUtils,
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

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);

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
  S: UnicodeString;
begin
  S := '';

  if Tokenizer = WesTokenizer then begin
    for i := 0 to High(Tokens) do
      S := S + Decode(Tokens[i]);
  end
  else
    S := DecodeGPT2Tokens(Tokens);

  Write(UTF8Encode(ConsoleText(S)));
end;

// Compute probability entropy in nats.
function ProbabilityEntropy(const Probs: array of Single; const Count: Integer): Double;
var
  i: Integer;
  P: Double;
begin
  Result := 0.0;

  for i := 0 to Count - 1 do begin
    P := Probs[i];

    if P > 0.0 then
      Result := Result - P * Ln(P);
  end;
end;

// Data reporting for inference.
procedure ReportInferenceDiagnostics(const Tok: Integer; const RawProb, AdjProb,
  SampleProb, Top1Prob, Top2Prob, TopKMass, Entropy, EOSProb: Double);
begin
  Write('Token = ', Tok, ' "', UTF8Encode(DecodeInferenceToken(Tok)), '"    RawP = ', RawProb: 0: 6, ' AdjP = ', AdjProb: 0: 6, ' SampP = ', SampleProb: 0: 6);
  Writeln('    Top1 = ', Top1Prob:0:6, ' Top2 = ', Top2Prob:0:6, ' Top5Mass = ', TopKMass:0:6, ' Entropy = ', Entropy:0:4, ' EOS = ', EOSProb:0:6);
end;

// Sample the top probs.
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
procedure InferOneToken(var WModelParams: TWModelParams; var WModelState: TWModelState; const Step: Integer;
  const QueryTokenized: TIVector; var QueryToken: Integer; var AdjustedProb: Single);
const
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).
  KSample = 5;
var
  j, Blk, LastPos, BestTok, TopTok: Integer;
  TopTokVector: array[0..KSample - 1] of Integer;
  TopProbVector: array[0..KSample - 1] of Single;
  ModelProb, TopKSampleProb: Single;
  RawProb, AdjProb: Single;
  RawEOSProb, Top1Prob, Top2Prob: Single;
  TopKMass, Entropy: Double;
  RawProbs: array of Single;

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
        Write('    ');
      end;

      Writeln;
    end;
    Top1Prob := TopProbVector[0];
    Top2Prob := TopProbVector[1];

    TopKMass := 0.0;
    for j := 0 to KSample - 1 do
      if TopProbVector[j] > 0.0 then
        TopKMass := TopKMass + TopProbVector[j];  end;

begin
  // Check for valid query.
  if Length(QueryTokenized) = 0 then begin
    QueryToken := EOS;
    AdjustedProb := 0.0;
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
      Write('Step ', Step, '. Context tokens = ', Length(QueryTokenized), '.');
      Write(' Last 20 InputTokens: ');
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

    SetLength(RawProbs, nVocab);
    for j := 0 to nVocab - 1 do
      RawProbs[j] := Probs[LastPos, j];
    Entropy := ProbabilityEntropy(RawProbs, nVocab);
    if (EOS >= 0) and (EOS < nVocab) then
      RawEOSProb := RawProbs[EOS]
    else
      RawEOSProb := 0.0;

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

    // Discourage immediate repetition before finding the top K.
    if Length(QueryTokenized) > 0 then
      Probs[LastPos, QueryTokenized[High(QueryTokenized)]] := Probs[LastPos, QueryTokenized[High(QueryTokenized)]] * 0.25;

    // Find the adjusted top-K candidates.
    TopKProbs;

    // Sample one token from the top-K candidates.
    BestTok := SampleTopK(TopTokVector, TopProbVector, TopTok, TopKSampleProb);

    RawProb := RawProbs[BestTok];
    AdjProb := Probs[LastPos, BestTok];

    QueryToken := BestTok;
    // QueryProb := AdjProb;

    if VerboseInfer then
      ReportInferenceDiagnostics(BestTok, RawProb, AdjProb, TopKSampleProb,
        Top1Prob, Top2Prob, TopKMass, Entropy, RawEOSProb);

    {// Raw model probability after special-token suppression and repetition penalty.
    ModelProb := Probs[LastPos, BestTok];

    // Return the selected token and its adjusted model probability.
    QueryToken := BestTok;
    AdjustedProb := ModelProb;}

  end;
end;

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
const
  MaxNewTokens = 500;
var
  i, Step, QueryToken: Integer;
  OldNVocab: Integer;
  OldTraining, OldVerboseTransform, OldSaveTokenizationFiles: Boolean;
  OwnsCuda, StartedCudaHere: Boolean;
  QueryTokenized, WorkTokens, QueryOutput: TIVector;
  AdjustedProb: Single;
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
      StartCuda(WModelParams, WModelState, WAdamWState);
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
        RunGPT2TokenizeString(QueryString, QueryTokenized);

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
        InferOneToken(WModelParams, WModelState, Step, WorkTokens, QueryToken, AdjustedProb);

        // Add the newly generated token to the output.
        SetLength(QueryOutput, Length(QueryOutput) + 1);
        QueryOutput[High(QueryOutput)] := QueryToken;

        // Stop after recording EOS.
        if QueryToken = EOS then Break;

        // EOS does not need to be placed back into the next input.
        SetLength(WorkTokens, Length(WorkTokens) + 1);
        WorkTokens[High(WorkTokens)] := QueryToken;

        Write('WorkTokens: <<');
        WriteInferenceTokens(WorkTokens);
        Writeln('>>');
        Pause;
      end;

      Write('Query decoded token output: ');
      WriteInferenceTokens(QueryOutput);
      Writeln;

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
        EndCuda(WModelParams, WModelState, WAdamWState);
    finally
      Training := OldTraining;
      VerboseTransform := OldVerboseTransform;
      SaveTokenizationFiles := OldSaveTokenizationFiles;
      nVocab := OldNVocab;
    end;
  end;
end;

end.

