unit Infer;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  GPT2Tokenize,
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

function CheckDLL(const LibName: string): Boolean;
var
  DLLHandle: THandle;
begin
  // Attempt to load the library.
  DLLHandle := LoadLibrary(PChar(LibName));

  // If the handle is non-zero, it loaded successfully.
  Result := (DLLHandle <> 0);

  // Clean up if it was successfully loaded.
  if Result then
    FreeLibrary(DLLHandle);
end;

// Check DLL accessibility.
procedure CheckAllDLLs;
begin
  If CheckDLL('cublas64_13.dll') then CublasPresent := True else CublasPresent := False;
  If CheckDLL('cudart64_13.dll') then CudartPresent := True else CudartPresent := False;
  If CheckDLL('WesChatKernel12.dll') then WesChatKernelPresent := True else WesChatKernelPresent := False;
  if not CublasPresent or not CudartPresent or not WesChatkernelPresent then begin
      Writeln('One of the following DLLs is required but not present: cublas64_13.dll, cudart64_13.dll, WesChatKernel12.dll.');
      Pause;
      Halt;
  end;
  if CublasPresent and (cublasCreate_v2(CuHandle) <> 0) then begin
    Writeln('cuBLAS initialization required but failed.');
    Pause;
    Halt;
  end;
end;

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
  const QueryTokenized: TIVector; var QueryToken: Integer);
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
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    // Build X only for block 0.
    // cublas.
    LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

    // Scale only block 0 input.
    // Optional transformer-style embedding scaling by sqrt(d_model).
    CuScale(CuHandle, SeqLen * ModelDim, Scale, WModelState.StateBlock[0].X.dValue);

    // Forward pass through stacked transformer blocks.
    for Blk := 0 to nBlock - 1 do begin
      Writeln('$$$ Inference Starting Block ', Blk, ' $$$');

      if VerboseTransform then begin
        cudaMemcpy(@StateBlock[Blk].X.Value[0, 0], StateBlock[Blk].X.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X.Value before transform.', StateBlock[Blk].X.Value, G);
      end;

      RunTransformForward(WModelParams, WModelState, Blk, 0);

      // Feed this block's output into the next block's input.
      if Blk < nBlock - 1 then
        // cblas.
        // CopyXTensor(WModelState.StateBlock[Blk].X7, WModelState.StateBlock[Blk + 1].X);
        // cuda kernel.
        cudaMemcpy(WModelState.StateBlock[Blk + 1].X.dValue, WModelState.StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    // Compute logit and probits.
    // cublas.
    LastBlk := nBlock - 1;
    CuMatMulFullNT(CuHandle, StateBlock[LastBlk].X7.dValue, WModelParams.Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

    // Softmax Forward.
    // cuda kernel.
    LaunchSoftmaxForwardN(dProbs, dProbs, SeqLen, nVocab, Temperature);

    // If nBlock, then pick the largest probs, and save them.
    cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
    Writeln('            Inference Stage');
    BestProb := Probs[LastPos, 0];
    BestTok := 0;
    for j := 1 to nVocab - 1 do
      if Probs[LastPos, j] > BestProb then begin
        BestProb := Probs[LastPos, j];
        BestTok := j;
      end;

    // BestTok is the next predicted token.
  end;

  QueryToken := BestTok;
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
  QueryInput: TBVector;
  QueryString: string;
begin
  CheckAllDLLs;
  Training := False;
  nVocab := nSymbols;

  InitializeTransformerState(WModelState);
  MAllocCublas(WModelParams, WModelState);

  try
    CopyParamsToDevice(WModelParams);
    CopyInvFreqToDevice(WModelState);

    Training := False;

    repeat
      Writeln('Enter query: ');
      Readln(QueryString);

      if QueryString = EmptyStr then Break;

      if VerboseTransform then with WModelParams do begin
        cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
      end;

      SetLength(QueryInput, Length(QueryString));
      for i := 0 to Length(QueryString) - 1 do
        QueryInput[i] := Ord(QueryString[i + 1]);

      if Tokenizer = WesTokenizer then
        RunWesTokenize(QueryInput, QueryTokenized)
      else
        RunGPT2Tokenize(QueryString, QueryTokenized);

      if Length(QueryTokenized) = 0 then begin
        Writeln('No tokens produced.');
        Continue;
      end;

      SetLength(QueryOutput, 0);
      WorkTokens := Copy(QueryTokenized);

      for Step := 1 to MaxNewTokens do begin
        InferOneToken(WModelParams, WModelState, WorkTokens, QueryToken);

        SetLength(QueryOutput, Length(QueryOutput) + 1);
        QueryOutput[High(QueryOutput)] := QueryToken;

        SetLength(WorkTokens, Length(WorkTokens) + 1);
        WorkTokens[High(WorkTokens)] := QueryToken;

        Writeln('Single Token Query Output: ', QueryToken);

        if QueryToken = EOS then Break;
      end;

      Writeln('Query token output: ');
      for i := 0 to High(QueryOutput) do
        Write(QueryOutput[i], ' ');
      Writeln;

      if Tokenizer = GPT2Tokenizer then begin
        // GPT2.
        LoadVocab('vocab.json', Vocab);
        Writeln('Query decoded token output: ');
        for i := 0 to High(QueryOutput) do
          Write(DisplayToken(UTF8Decode(Vocab[QueryOutput[i]])));
      end
      else begin
        // WesChat.
        Writeln('Query decoded token output: ');
        DetokenizeToDisplay(QueryOutput, F);
      end;
      Writeln;
    until False;

  finally
    MDeallocateCublas(WModelParams, WModelState);
    cublasDestroy_v2(CuHandle);
  end;
end;


end.

