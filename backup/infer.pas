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
  SysUtils,
  TransformForward,
  WesTokenize,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState);

implementation

// Build the input vector .
{procedure BuildInputVector(var Input: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Input[i] := TokenizedCorpus[StartIndex + i];
end;}

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
var
  j, Blk, LastBlk, LastPos: Integer;
  BestTok: Integer;
  BestProb: Single;
begin
  // Check for valid query.
  if Length(QueryTokenized) = 0 then begin
    QueryToken := EOS;
    Exit;
  end;

  with WModelParams do
    if VerboseTransform then begin
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
    end;

  with WModelState do begin
    // Build the input vector.
    BuildInferenceInputTokens(InputTokens, QueryTokenized, SeqLen, LastPos);

    Writeln('Infer step Length(QueryTokenized)=', Length(QueryTokenized), ' LastPos=', LastPos);
    Write('Last few InputTokens: ');
    for j := Max(0, LastPos - 5) to LastPos do
      Write(InputTokens[j], ' ');
    Writeln;

    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    for j := 0 to Length(QueryTokenized) - 1 do
      if (InputTokens[j] < 0) or (InputTokens[j] >= nVocab) then
        Writeln('BAD TOKEN at ', j, ': ', InputTokens[j], ' nVocab=', nVocab);

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

      RunTransformForward(WModelParams, WModelState, Blk, 0);

      // Feed this block's output into the next block's input.
      if Blk < nBlock - 1 then
        // CopyXTensor(WModelState.StateBlock[Blk].X7, WModelState.StateBlock[Blk + 1].X);
        cudaMemcpy(WModelState.StateBlock[Blk + 1].X.dValue, WModelState.StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    // Compute logit and probits.
    LastBlk := nBlock - 1;
    CuMatMulFullNT(CuHandle, StateBlock[LastBlk].X7.dValue, WModelParams.Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

    // Softmax Forward.
    LaunchSoftmaxForwardN(dProbs, dProbs, SeqLen, nVocab, Temperature);
    cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);

    // Check that probs are changing.
    for j := 0 to 4 do
      Write('tok ', j, ' prob ', Probs[LastPos, j]: 6: 6, '   ');
    Pause;

    // If nBlock, then pick the largest probs, and save them.
    Writeln('Final Inference Stage, pick largest prob');
    BestProb := Probs[LastPos, 0];
    BestTok := 0;
    for j := 1 to nVocab - 1 do
      if Probs[LastPos, j] > BestProb then begin
        BestProb := Probs[LastPos, j];
        BestTok := j;
      end;

    // BestTok is the next predicted token.
    QueryToken := BestTok;
    QueryProb := BestProb;

    {Write('All the probs in lastpos: ');
    for j := 1 to nVocab - 1 do
      Write(j: 3, ' ', Probs[LastPos, j]: 6: 6,  ' ');
    Writeln;}
  end;
end;

// Run inference forward without additional training.
// QueryInput is the query from the user, like Corpus.
// QueryString is the string input from the user.
// QueryOutput is the output tokens from the model.
// QueryTokenized is the tokenization of QueryInput, like TokenizedCorpus.
// QueryToken is the single next token produced by the infer proc.
procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState);
const
  MaxNewTokens = 100;        // Limit on new tokens produced.
var
  i, Step, QueryToken: Integer;
  QueryTokenized, WorkTokens, QueryOutput: TIVector;
  QueryProb: Single;
  QueryInput: TBVector;
  QueryString: string;
begin
  Training := False;
  nVocab := nSymbols;
  VerboseTransform := False;         // Select verbosity during inference.

  // Initialize transformer state.
  InitializeTransformerState(WModelState);

  // Intialize cublas and cuda and malloc.
  if not CuBlasInitialized then
    InitializeCuBlas;
  if not CudaAllocated then
    MAllocCublas(WModelParams, WModelState);

  try
    // Send params (if new model) and inverse freq to device.
    if ParamsNeedCopyToDevice then
      CopyParamsToDevice(WModelParams);
    CopyInvFreqToDevice(WModelState);

    begin // Run one query. (Later, repeat loop.)
      // Get a query from user.
      // Write('Enter query: ');
      // Readln(QueryString);
      QueryString := 'man with';   // Temporary.
      Writeln('Query string: ', QueryString);

      if QueryString = EmptyStr then begin
        Writeln('Query is empty.');
        Exit;
      end;

      SetLength(QueryInput, Length(QueryString));
      for i := 0 to Length(QueryString) - 1 do
        QueryInput[i] := Ord(QueryString[i + 1]);

      Write(Length(QueryInput), ' ', 'Query Tokens (Query String as Tokens): ');
      for i := 0 to Length(QueryInput) - 1 do
         Write(QueryInput[i], ' ');
      Writeln;

      if Tokenizer = WesTokenizer then
        RunWesTokenize(QueryInput, QueryTokenized)
      else
        RunGPT2Tokenize(QueryString, QueryTokenized);

      if Length(QueryTokenized) = 0 then begin
        Writeln('No tokens produced.');
        Exit;
      end;

      if (Length(QueryTokenized) > 0) and (QueryTokenized[High(QueryTokenized)] = EOS) then
        SetLength(QueryTokenized, Length(QueryTokenized) - 1);
      Writeln('Last token after removal = ',
              QueryTokenized[High(QueryTokenized)]);
      Writeln('EOS=', EOS,
              ' Last=', QueryTokenized[High(QueryTokenized)]);

      Write('Query Tokens, Again, Using TCFull');
      TCFull(QueryTokenized);

      SetLength(QueryOutput, 0);
      WorkTokens := Copy(QueryTokenized);

      for Step := 1 to MaxNewTokens do begin
        // Run Infer to get one additional token.
        InferOneToken(WModelParams, WModelState, WorkTokens, QueryToken, QueryProb);

        // QueryToken is new token, add it to original query, in WorkTokens.
        SetLength(QueryOutput, Length(QueryOutput) + 1);
        QueryOutput[High(QueryOutput)] := QueryToken;
        SetLength(WorkTokens, Length(WorkTokens) + 1);
        WorkTokens[High(WorkTokens)] := QueryToken;

        //if Tokenizer = WesTokenizer then
          // WesTokenizer.
          Writeln('Single Query Token Number = ', QueryToken, ' Probability = ', QueryProb: 6: 6,
          ' Decoded = ', Decode(QueryToken));
        //else
          // GPT2Tokenizer.
          //Writeln('Single Query Token Number = ', QueryToken, ' Probability = ', WModelState.Probs[SeqLen - 1, QueryToken]: 6: 6,
          //' Decoded = ', DisplayToken(UTF8Decode(Vocab[QueryToken])));

          Write('Cumulative Decode: ');
          for i := 0 to High(QueryOutput) do
            // WesTokenizer.
            // Write(SymbolTable[QueryOutput[i]]);
            Write(Decode(QueryOutput[i]));
          Writeln;

          Write('New Query Input, WorkTokens: ');
          for i := 0 to High(WorkTokens) do
            Write(Decode(WorkTokens[i]));
          Writeln;
        Pause;

        if QueryToken = EOS then Break;
      end;

      // Add new token to string QueryOutput.
      if Length(QueryOutput) > 0 then
        QueryToken := QueryOutput[High(QueryOutput)];

      Writeln('Query token output: ');
      for i := 0 to High(QueryOutput) do
        Write(QueryOutput[i], ' ');
      Writeln;

      if Tokenizer = GPT2Tokenizer then begin
        // GPT2.
        Write('Query decoded token output: ');
        for i := 0 to High(QueryOutput) do
          if Assigned(Vocab) and (QueryOutput[i] >= 0) and (QueryOutput[i] < Vocab.Count) then
            Write(DisplayToken(UTF8Decode(Vocab[QueryOutput[i]])))
          else
            Write('<BADTOKEN:', QueryOutput[i], '>');
      end
      else begin
        // WesChat.
        Write('Query decoded token output: ');
        DetokenizeToDisplay(QueryOutput, F);
      end;
      Writeln;
    end; // Run query once.
  finally
    Training := True;
    // MDeallocateCublas(WModelParams, WModelState);
    // cublasDestroy_v2(CuHandle);
  end;
end;


end.

