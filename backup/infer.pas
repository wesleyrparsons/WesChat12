unit Infer;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Matrix,
  SysUtils,
  TransformForward,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const QueryTokenized: TIVector; var QueryOutput: TIVector);

implementation

const
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).

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
procedure BuildInputVector(var Input: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Input[i] := TokenizedCorpus[StartIndex + i];
end;

procedure InferOneToken(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const QueryTokenized: TIVector; var QueryToken: Integer);
var
  i, j, Blk, LastBlk, LastPos: Integer;
  Start, EmbedLoop: Integer;
  BestTok: Integer;
  BestProb: Single;
  Stride: Integer = 64;      // Stride 64 tokens every sequence.

begin

  with WModelParams do
    if VerboseTransform then begin
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
    end;

  // Initialize.
  // PromptLen := Length(QueryTokenized);

  // Stride loop thru Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(QueryTokenized) do with WModelState do begin

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    Writeln('&&& Loop thru Embed: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    Writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Build the input vector.
    BuildInputVector(InputTokens, QueryTokenized, Start, SeqLen);
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    // Build X only for block 0.
    // cublas.
    LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

    // Scale only block 0 input.
    // Optional transformer-style embedding scaling by sqrt(d_model).
    CuScale(CuHandle, SeqLen * ModelDim, Scale, WModelState.StateBlock[0].X.dValue);

    // Forward pass through stacked transformer blocks.
    for Blk := 0 to nBlock - 1 do begin
      Writeln('$$$ Starting Block ', Blk, '  Sequence Start ', Start, ' $$$');

      if VerboseTransform then begin
        cudaMemcpy(@StateBlock[Blk].X.Value[0, 0], StateBlock[Blk].X.dValue, XSize, cudaMemcpyDeviceToHost);
        VTPDisplayX('Display X.Value before transform.', StateBlock[Blk].X.Value, G);
      end;

      RunTransformForward(WModelParams, WModelState, Blk);

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
    LastPos := SeqLen - 1;
    for i := 0 to SeqLen - 1 do begin
      BestProb := Probs[LastPos, 0];
      BestTok := 0;

      for j := 1 to nVocab - 1 do
        if Probs[LastPos, j] > BestProb then begin
          BestProb := Probs[LastPos, j];
          BestTok := j;
        end;
    end;

    // BestTok is the next predicted token.
    Start := Start + Stride;
  end;

  QueryToken := BestTok;
end;

// Run inference forward without additional training.
procedure RunInfer(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const QueryTokenized: TIVector; var QueryOutput: TIVector);
const
  EOS = 257;
var
  k, QueryToken: Integer;
  Start: Integer;
  // Stride: Integer = 64;      // Stride 64 tokens every sequence.

begin

  CheckAllDLLs;

  // Initialize.
  InitializeTransformer(WModelParams, WModelState);
  MAllocCublas(WModelParams, WModelState);
  CopyParamsToDevice(WModelParams);
  CopyInvFreqToDevice(WModelState);

  Writeln('First quarter of one row of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModelParams.Embeddings.Value[1, k]: 8: 6, ' ');
  Writeln;
  Pause;

  with WModelParams do
    if VerboseTransform then begin
      cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
      VTPDisplayX('Display Embeddings.Value prior to Transform.', Embeddings.Value, B);
    end;

  repeat
    InferOneToken(WModelParams, WModelState, QueryTokenized, QueryToken);
    SetLength(QueryOutput, Length(QueryOutput) + 1);
    QueryOutput[Length(QueryOutput)] := QueryToken;
    Writeln('Single Token Query Output: ', QueryToken);
  until QueryToken = EOS;
end;

end.

