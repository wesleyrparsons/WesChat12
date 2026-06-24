unit Util;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  Math,
  Matrix,
  SysUtils;

const
  WeightSize: Integer = ModelDim * ModelDim * SizeOf(Single);
  WeightProjectedSize: Integer = ModelDim * ModelDimProj * SizeOf(Single);
  ProjectedSize: Integer = ModelDimProj * SizeOf(Single);
  SeqSize: Integer = SeqLen * SizeOf(Single);
  ModelSize: Integer = ModelDim * SizeOf(Single);
  XSize: Integer = SeqLen * ModelDim * SizeOf(Single);
  HiddenSize: Integer = SeqLen * ModelDimProj * SizeOf(Single);
  ScoresSize: Integer = SeqLen * SeqLen * SizeOf(Single);
  EmbeddingsSize: Integer = DimVocab * ModelDim * SizeOf(Single);
  InvFreqSize: Integer = (ModelDim div 2) * SizeOf(Single);
  ProbsSize: Integer = SeqLen * DimVocab * SizeOf(Single);

// Cublas and Cuda procedures.
procedure InitializeCublas;
procedure CheckCudaError(const Where: string);
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState);
// Word procedures and functions.
procedure InitWorkFolders(const Root: string);
function CleanBaseName(const FileName: string): string;
function WorkSymbolFile(const BaseName: string): string;
function WorkTokenFile(const BaseName: string): string;
function WorkModelFile(const BaseName: string): string;
function WorkLogFile(const BaseName: string): string;
function WorkRunFile(const BaseName: string): string;
// Utility procedures.
procedure PadToSeqMultiple(var TokenVectorToPad: TIVector; const Seq: Integer);
procedure TC100(const TC: TIVector);
procedure TCFull(const TC: TIVector);
function Decode(const x: Integer): UnicodeString;
// Transform routines.
function ComputeLoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;
procedure ScaleAllGradients(var WModelParams: TWModelParams; const S: Single);
// Initialization routines.
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
procedure InitializeTransformerState(var WModelState: TWModelState);
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
procedure CopyParamsToHost(var WModelParams: TWModelParams);
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
// Optimization routines.
//procedure CuUpdateParam(Handle: TcublasHandle; const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
procedure Optimization(var WModelParams: TWModelParams; const Blk: Integer);
procedure UpdateEmbeddings(var WModelParams: TWModelParams; var WModelState: TWModelState);
// DLL transform routines.
procedure LaunchClipVector(X: PSingle; N: Integer; Limit: Single); cdecl;
  external 'WesChatKernel12.dll';
procedure LaunchEmbeddingLookup(Embeddings: PSingle; InputTokens: PInteger; X: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAutoRegressiveMask(Scores: PSingle; SeqLen: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAutoRegressiveMaskBackward(ScoresGrad: PSingle; SeqLen: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchDropout(X: PSingle; N: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchDropoutBackward(dX: PSingle; N: Integer; DropProb: Single; Seed: UInt64);
  cdecl; external 'WesChatKernel12.dll';procedure SoftmaxForwardN(const x: PSingle; y: PSingle; const N: Integer);
procedure LaunchSoftmaxForwardStrided(dIn: PSingle; dOut: PSingle; Rows: Integer; Cols: Integer; RowStride: Integer; Temperature: Single);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchSoftmaxForwardN(X: PSingle; Y: PSingle; Rows: Integer; N: Integer; Temperature: Single);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchSoftmaxBackward(Y: PSingle; dY: PSingle; dX: PSingle; Rows: Integer; D: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchLayerNormForward(InX, OutX, Gamma, Beta, LNXhat, LNInvStd: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchLayerNormBackward(dY, dX, Gamma, LNXhat, LNInvStd, dGamma, dBeta: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchRoPEForward(H: PSingle; InvFreq: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchRoPEBackward(dH: PSingle; InvFreq: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchCEGradient(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; SeqLen: Integer; nVocab: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchCEGradientStrided(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; Rows: Integer; VocabSize: Integer; RowStride: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddInputEmbeddingGrad(XGrad: PSingle; EmbGrad: PSingle; InputTokens: PInteger; SeqLen: Integer; ModelDim: Integer; nVocab: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddBiasRows(X: PSingle; Bias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddBiasRowsBackward(dX: PSingle; dBias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';

implementation

// Check existence of any DLL.
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

// Check accessibility of three necessary DLLs.
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
end;

// Intialize Cublas.
procedure InitializeCublas;
begin
  CheckAllDLLs;
  if Cublas_Init then
      Writeln('CuBLAS successfully initiated.')
  else begin
    Writeln('Error initiating CuBLAS. Halting....');
    Pause;
    Halt;
  end;
end;

// Check for a cuda error.
procedure CheckCudaError(const Where: string);
var
  Err: Integer;
begin
  Err := cudaDeviceSynchronize;

  if Err <> 0 then begin
    Writeln;
    Writeln('*** CUDA ERROR ***');
    Writeln('Location : ', Where,' ', 'Error #  : ', Err, ' ', 'Message  : ', StrPas(cudaGetErrorString(Err)));
    Pause;
  end;
end;

// Intialize Cuda and Cublas.
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  CheckAllDLLs;
  InitializeCublas;
  if not CudaAllocated then
    MAllocCublas(WModelParams, WModelState);
  CheckCudaError('Start cuda.');
end;

// End Cuda and Cublas.
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState);
  if CuBLAS_Shutdown then
    Writeln('CuBLAS successfully shut down.')
end;

// Saving routines.
procedure InitWorkFolders(const Root: string);
begin
  WorkRoot := IncludeTrailingPathDelimiter(ExpandFileName(Root));

  CorpusDir  := WorkRoot + 'corpus'  + DirectorySeparator;
  SymbolDir  := WorkRoot + 'symbols' + DirectorySeparator;
  TokenDir   := WorkRoot + 'tokens'  + DirectorySeparator;
  ModelDir   := WorkRoot + 'models'  + DirectorySeparator;
  LogDir     := WorkRoot + 'logs'    + DirectorySeparator;
  RunDir     := WorkRoot + 'runs'    + DirectorySeparator;
  ScratchDir := WorkRoot + 'scratch' + DirectorySeparator;

  ForceDirectories(WorkRoot);
  ForceDirectories(CorpusDir);
  ForceDirectories(SymbolDir);
  ForceDirectories(TokenDir);
  ForceDirectories(ModelDir);
  ForceDirectories(LogDir);
  ForceDirectories(RunDir);
  ForceDirectories(ScratchDir);
end;

function CleanBaseName(const FileName: string): string;
begin
  Result := ChangeFileExt(ExtractFileName(FileName), '');
end;

function WorkSymbolFile(const BaseName: string): string;
begin
  Result := SymbolDir + ChangeFileExt(ExtractFileName(BaseName), '.sym');
end;

function WorkTokenFile(const BaseName: string): string;
begin
  Result := TokenDir + ChangeFileExt(ExtractFileName(BaseName), '.tok');
end;

function WorkModelFile(const BaseName: string): string;
begin
  Result := ModelDir + ChangeFileExt(ExtractFileName(BaseName), '.model');
end;

function WorkLogFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(ExtractFileName(BaseName), '.log');
end;

function WorkRunFile(const BaseName: string): string;
begin
  Result := RunDir + BaseName + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.run';
end;

// Pad token vector to multiple of SeqLen.
procedure PadToSeqMultiple(var TokenVectorToPad: TIVector; const Seq: Integer);
var
  OldLen, NewLen, i: Integer;
begin
  OldLen := Length(TokenVectorToPad);
  NewLen := ((OldLen + Seq - 1) div Seq) * Seq;

  SetLength(TokenVectorToPad, NewLen);

  for i := OldLen to NewLen - 1 do
    TokenVectorToPad[i] := PAD;
end;

// Display first 100 tokens of tokenized corpus and detokenized form.
procedure TC100(const TC: TIVector);
var
  i: Integer;
begin
  Write('Tokenized Corpus (length up to 100): ');
  for i := 0 to Min(99, High(TC)) do
    Write(TC[i], ' ');
  Writeln;
  Write('Detokenized Corpus (length up to 100): ');
  for i := 0 to Min(99, High(TC)) do
    Write(SymbolTable[TC[i]]);
  Writeln;
end;

// Display tokenized corpus tokens and detokenized corpus (as check).
procedure TCFull(const TC: TIVector);
var
  i: Integer;
begin
  Write('Tokenized Corpus (in full length of ', Length(TC), '): ');
  for i := 0 to High(TC) do
    Write(TC[i], ' ');
  Writeln;

  Write('Detokenized Corpus (in full length of ', Length(TC) - 1, '): ');
  for i := 0 to High(TC) do
    Write(SymbolTable[TC[i]]);
  Writeln;
  Pause;
end;

// Decode using symbol table one token.
function Decode(const x: Integer): UnicodeString;
begin
  if Tokenizer = WesTokenizer then
  // WesTokenizer.
  Result := SymbolTable[x]
else
  // GPT2Tokenizer.
  Result := UTF8Decode(Vocab[x]);
end;

// Compute cross-entropy loss.
function ComputeLoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;
const
  Eps = 1.0e-12;
var
  i: Integer;
  P: Double;
begin
  Result := 0.0;

  for i := 0 to SeqLen - 1 do begin
    P := Probs[i, TargetTokens[i]];

    if P < Eps then
      P := Eps;

    Result := Result - Ln(P);
  end;

  Result := Result / SeqLen;
end;

// Scale all the gradients.
procedure ScaleAllGradients(var WModelParams: TWModelParams; const S: Single);
var
  k: Integer;
begin
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin
      CuScale(CuHandle, ModelDim * ModelDim, S, Wq.dGrad);
      CuScale(CuHandle, ModelDim * ModelDim, S, Wk.dGrad);
      CuScale(CuHandle, ModelDim * ModelDim, S, Wv.dGrad);
      CuScale(CuHandle, ModelDim * ModelDim, S, W0.dGrad);

      CuScale(CuHandle, ModelDim * ModelDimProj, S, W1.dGrad);
      CuScale(CuHandle, ModelDimProj * ModelDim, S, W2.dGrad);

      CuScale(CuHandle, ModelDimProj, S, b1.dGrad);
      CuScale(CuHandle, ModelDim,     S, b2.dGrad);

      CuScale(CuHandle, ModelDim, S, Gamma1.dGrad);
      CuScale(CuHandle, ModelDim, S, Beta1.dGrad);
      CuScale(CuHandle, ModelDim, S, Gamma2.dGrad);
      CuScale(CuHandle, ModelDim, S, Beta2.dGrad);
    end;

  CuScale(CuHandle, DimVocab * ModelDim, S, WModelParams.Embeddings.dGrad);
end;

// Xavier-Glorot initialization on W0 matrix.
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on WHead matrix.
procedure XGUniformWHead(var W: TWeightHeadMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to HeadDim - 1 do
    for j := 0 to HeadDim - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W1 matrix.
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[i, j] := (2 * r - 1) * Limit;
    end;
end;

// Xavier-Glorot initialization on W2 matrix.
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
var
  Limit, r: Single;
  i, j: Integer;
begin
  Limit := Sqrt(6.0 / (FanIn + FanOut));

  for i := 0 to ModelDim - 1 do
    for j := 0 to ModelDimProj - 1 do begin
      r := Random;              // 0..1.
      W[j, i] := (2 * r - 1) * Limit;
    end;
end;

// Allocate cublas memory.
// Separate for State and Params? Not necessary.
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  h, k: Integer;
begin
  CudaAllocated := True;

  // Input and target tokens.
  cudaMalloc(@dInputTokens, SeqLen * SizeOf(Integer));
  cudaMalloc(@dTargetTokens, SeqLen * SizeOf(Integer));

  // Global/shared parameters.
  with WModelParams do begin
    cudaMalloc(@Embeddings.dValue, EmbeddingsSize);
    cudaMalloc(@Embeddings.dGrad, EmbeddingsSize);
  end;
  with WModelState do begin
    cudaMalloc(@dInvFreq, InvFreqSize);
    cudaMalloc(@dProbs, ProbsSize);
    cudaMalloc(@dTopGradient, ProbsSize);
  end;

  // Per block parameters.
  for k := 0 to nBlock - 1 do begin
    with WModelParams.ParamBlock[k] do begin
      cudaMalloc(@Wq.dValue, WeightSize);
      cudaMalloc(@Wq.dGrad, WeightSize);
      cudaMalloc(@Wk.dValue, WeightSize);
      cudaMalloc(@Wk.dGrad, WeightSize);
      cudaMalloc(@Wv.dValue, WeightSize);
      cudaMalloc(@Wv.dGrad, WeightSize);
      cudaMalloc(@W0.dValue, WeightSize);
      cudaMalloc(@W0.dGrad, WeightSize);
      cudaMalloc(@W1.dValue, WeightProjectedSize);
      cudaMalloc(@W1.dGrad, WeightProjectedSize);
      cudaMalloc(@W2.dValue, WeightProjectedSize);
      cudaMalloc(@W2.dGrad, WeightProjectedSize);
      cudaMalloc(@b1.dValue, ProjectedSize);
      cudaMalloc(@b1.dGrad, ProjectedSize);
      cudaMalloc(@b2.dValue, ModelSize);
      cudaMalloc(@b2.dGrad, ModelSize);
      cudaMalloc(@Gamma1.dValue, ModelSize);
      cudaMalloc(@Gamma1.dGrad, ModelSize);
      cudaMalloc(@Beta1.dValue, ModelSize);
      cudaMalloc(@Beta1.dGrad, ModelSize);
      cudaMalloc(@Gamma2.dValue, ModelSize);
      cudaMalloc(@Gamma2.dGrad, ModelSize);
      cudaMalloc(@Beta2.dValue, ModelSize);
      cudaMalloc(@Beta2.dGrad, ModelSize);
    end;
    with WModelState.StateBlock[k] do begin
      cudaMalloc(@X.dValue, XSize);
      cudaMalloc(@X.dGrad, XSize);
      cudaMalloc(@X1.dValue, XSize);
      cudaMalloc(@X1.dGrad, XSize);
      cudaMalloc(@X2.dValue, XSize);
      cudaMalloc(@X2.dGrad, XSize);
      cudaMalloc(@X3.dValue, XSize);
      cudaMalloc(@X3.dGrad, XSize);
      cudaMalloc(@X4.dValue, XSize);
      cudaMalloc(@X4.dGrad, XSize);
      cudaMalloc(@X5.dValue, XSize);
      cudaMalloc(@X5.dGrad, XSize);
      cudaMalloc(@X6.dValue, XSize);
      cudaMalloc(@X6.dGrad, XSize);
      cudaMalloc(@X7.dValue, XSize);
      cudaMalloc(@X7.dGrad, XSize);
      cudaMalloc(@X1q.dValue, XSize);
      cudaMalloc(@X1q.dGrad, XSize);
      cudaMalloc(@X1k.dValue, XSize);
      cudaMalloc(@X1k.dGrad, XSize);
      cudaMalloc(@X1v.dValue, XSize);
      cudaMalloc(@X1v.dGrad, XSize);
      cudaMalloc(@Q.dValue, XSize);
      cudaMalloc(@Q.dGrad, XSize);
      cudaMalloc(@K.dValue, XSize);
      cudaMalloc(@K.dGrad, XSize);
      cudaMalloc(@V.dValue, XSize);
      cudaMalloc(@V.dGrad, XSize);
      for h := 0 to nHead - 1 do begin
        cudaMalloc(@ScoresHead1[h].dValue, ScoresSize);
        cudaMalloc(@ScoresHead1[h].dGrad, ScoresSize);
        cudaMalloc(@ScoresHead2[h].dValue, ScoresSize);
        cudaMalloc(@ScoresHead2[h].dGrad, ScoresSize);
      end;
      cudaMalloc(@Hidden1.dValue, HiddenSize);
      cudaMalloc(@Hidden1.dGrad, HiddenSize);
      cudaMalloc(@Hidden2.dValue, HiddenSize);
      cudaMalloc(@Hidden2.dGrad, HiddenSize);
      cudaMalloc(@dLNInvStd1, SeqSize);
      cudaMalloc(@dLNXHat1, XSize);
      cudaMalloc(@dLNInvStd2, SeqSize);
      cudaMalloc(@dLNXHat2, XSize);
      cudaMalloc(@dX4FromLN2, XSize);
      cudaMalloc(@dXFromLN1, XSize);
    end;
  end;
end;

// De-allocate cublas memory.
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState);
var
  h, k: Integer;
begin
  CudaAllocated := False;

  cudaFree(dInputTokens);  dInputTokens  := nil;
  cudaFree(dTargetTokens); dTargetTokens := nil;

  with WModelParams do begin
    cudaFree(Embeddings.dValue); Embeddings.dValue := nil;
    cudaFree(Embeddings.dGrad);  Embeddings.dGrad  := nil;
  end;

  with WModelState do begin
    cudaFree(dInvFreq);     dInvFreq     := nil;
    cudaFree(dProbs);       dProbs       := nil;
    cudaFree(dTopGradient); dTopGradient := nil;
  end;

  for k := 0 to nBlock - 1 do begin
    with WModelParams.ParamBlock[k] do begin
      cudaFree(Wq.dValue);     Wq.dValue     := nil;
      cudaFree(Wq.dGrad);      Wq.dGrad      := nil;
      cudaFree(Wk.dValue);     Wk.dValue     := nil;
      cudaFree(Wk.dGrad);      Wk.dGrad      := nil;
      cudaFree(Wv.dValue);     Wv.dValue     := nil;
      cudaFree(Wv.dGrad);      Wv.dGrad      := nil;
      cudaFree(W0.dValue);     W0.dValue     := nil;
      cudaFree(W0.dGrad);      W0.dGrad      := nil;
      cudaFree(W1.dValue);     W1.dValue     := nil;
      cudaFree(W1.dGrad);      W1.dGrad      := nil;
      cudaFree(W2.dValue);     W2.dValue     := nil;
      cudaFree(W2.dGrad);      W2.dGrad      := nil;
      cudaFree(b1.dValue);     b1.dValue     := nil;
      cudaFree(b1.dGrad);      b1.dGrad      := nil;
      cudaFree(b2.dValue);     b2.dValue     := nil;
      cudaFree(b2.dGrad);      b2.dGrad      := nil;
      cudaFree(Gamma1.dValue); Gamma1.dValue := nil;
      cudaFree(Gamma1.dGrad);  Gamma1.dGrad  := nil;
      cudaFree(Beta1.dValue);  Beta1.dValue  := nil;
      cudaFree(Beta1.dGrad);   Beta1.dGrad   := nil;
      cudaFree(Gamma2.dValue); Gamma2.dValue := nil;
      cudaFree(Gamma2.dGrad);  Gamma2.dGrad  := nil;
      cudaFree(Beta2.dValue);  Beta2.dValue  := nil;
      cudaFree(Beta2.dGrad);   Beta2.dGrad   := nil;
    end;

    with WModelState.StateBlock[k] do begin
      cudaFree(X.dValue);   X.dValue   := nil;
      cudaFree(X.dGrad);    X.dGrad    := nil;
      cudaFree(X1.dValue);  X1.dValue  := nil;
      cudaFree(X1.dGrad);   X1.dGrad   := nil;
      cudaFree(X2.dValue);  X2.dValue  := nil;
      cudaFree(X2.dGrad);   X2.dGrad   := nil;
      cudaFree(X3.dValue);  X3.dValue  := nil;
      cudaFree(X3.dGrad);   X3.dGrad   := nil;
      cudaFree(X4.dValue);  X4.dValue  := nil;
      cudaFree(X4.dGrad);   X4.dGrad   := nil;
      cudaFree(X5.dValue);  X5.dValue  := nil;
      cudaFree(X5.dGrad);   X5.dGrad   := nil;
      cudaFree(X6.dValue);  X6.dValue  := nil;
      cudaFree(X6.dGrad);   X6.dGrad   := nil;
      cudaFree(X7.dValue);  X7.dValue  := nil;
      cudaFree(X7.dGrad);   X7.dGrad   := nil;

      cudaFree(X1q.dValue); X1q.dValue := nil;
      cudaFree(X1q.dGrad);  X1q.dGrad  := nil;
      cudaFree(X1k.dValue); X1k.dValue := nil;
      cudaFree(X1k.dGrad);  X1k.dGrad  := nil;
      cudaFree(X1v.dValue); X1v.dValue := nil;
      cudaFree(X1v.dGrad);  X1v.dGrad  := nil;

      cudaFree(Q.dValue);   Q.dValue   := nil;
      cudaFree(Q.dGrad);    Q.dGrad    := nil;
      cudaFree(K.dValue);   K.dValue   := nil;
      cudaFree(K.dGrad);    K.dGrad    := nil;
      cudaFree(V.dValue);   V.dValue   := nil;
      cudaFree(V.dGrad);    V.dGrad    := nil;

      for h := 0 to nHead - 1 do begin
        cudaFree(ScoresHead1[h].dValue); ScoresHead1[h].dValue := nil;
        cudaFree(ScoresHead1[h].dGrad);  ScoresHead1[h].dGrad  := nil;
        cudaFree(ScoresHead2[h].dValue); ScoresHead2[h].dValue := nil;
        cudaFree(ScoresHead2[h].dGrad);  ScoresHead2[h].dGrad  := nil;
      end;

      cudaFree(Hidden1.dValue); Hidden1.dValue := nil;
      cudaFree(Hidden1.dGrad);  Hidden1.dGrad  := nil;
      cudaFree(Hidden2.dValue); Hidden2.dValue := nil;
      cudaFree(Hidden2.dGrad);  Hidden2.dGrad  := nil;

      cudaFree(dLNInvStd1); dLNInvStd1 := nil;
      cudaFree(dLNXHat1);   dLNXHat1   := nil;
      cudaFree(dLNInvStd2); dLNInvStd2 := nil;
      cudaFree(dLNXHat2);   dLNXHat2   := nil;

      cudaFree(dX4FromLN2);
      cudaFree(dXFromLN1);
      dX4FromLN2 := nil;
      dXFromLN1  := nil;
    end;
  end;
end;

// Copy parameters from host to device.
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
var
  k: Integer;
begin
  // Embeddings (global).
  cudaMemcpy(WModelParams.Embeddings.dValue, @WModelParams.Embeddings.Value[0,0], EmbeddingsSize, cudaMemcpyHostToDevice);

  // Params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin
      // Attention weights.
      cudaMemcpy(Wq.dValue, @Wq.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wk.dValue, @Wk.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wv.dValue, @Wv.Value[0,0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W0.dValue, @W0.Value[0,0], WeightSize, cudaMemcpyHostToDevice);

      // MLP.
      cudaMemcpy(W1.dValue, @W1.Value[0,0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dValue, @W2.Value[0,0], WeightProjectedSize, cudaMemcpyHostToDevice);

      // Biases.
      cudaMemcpy(b1.dValue, @b1.Value[0], ProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b2.dValue, @b2.Value[0], ModelSize, cudaMemcpyHostToDevice);

      // LayerNorm.
      cudaMemcpy(Gamma1.dValue, @Gamma1.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta1.dValue,  @Beta1.Value[0],  ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma2.dValue, @Gamma2.Value[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta2.dValue,  @Beta2.Value[0],  ModelSize, cudaMemcpyHostToDevice);
  end;
end;

// Copy parameters from device to host (for saving model).
procedure CopyParamsToHost(var WModelParams: TWModelParams);
var
  k: Integer;
begin
  // Embeddings (global).
  cudaMemcpy(@WModelParams.Embeddings.Value[0,0], WModelParams.Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);

  // Params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin
      // Attention weights.
      cudaMemcpy(@Wq.Value[0,0], Wq.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wk.Value[0,0], Wk.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wv.Value[0,0], Wv.dValue, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W0.Value[0,0], W0.dValue, WeightSize, cudaMemcpyDeviceToHost);

      // MLP.
      cudaMemcpy(@W1.Value[0,0], W1.dValue, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W2.Value[0,0], W2.dValue, WeightProjectedSize, cudaMemcpyDeviceToHost);

      // Biases.
      cudaMemcpy(@b1.Value[0], b1.dValue, ProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b2.Value[0], b2.dValue, ModelSize, cudaMemcpyDeviceToHost);

      // LayerNorm.
      cudaMemcpy(@Gamma1.Value[0], Gamma1.dValue, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta1.Value[0],  Beta1.dValue,  ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma2.Value[0], Gamma2.dValue, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta2.Value[0],  Beta2.dValue,  ModelSize, cudaMemcpyDeviceToHost);
  end;
end;

// Copy InvFreq from host to device.
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
begin
  cudaMemcpy(WModelState.dInvFreq, @WModelState.InvFreq[0], InvFreqSize, cudaMemcpyHostToDevice);
end;

// Initialize the transformer state.
procedure InitializeTransformerState(var WModelState: TWModelState);
var
  j: Integer;
begin
  // Do not zero the state grads -- thet is done in zero gradient.
  // Do not zero the state values -- that is not necessary.

  // Compute InvFreq.
  SetLength(WModelState.InvFreq, ModelDim div 2);
  for j := 0 to (ModelDim div 2) - 1 do     // ModelDim must be even.
    WModelState.InvFreq[j] := Exp( - (2.0 * j) / ModelDim * Ln(10000.0) );
end;

// Initialize the transformer state stage.
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
var
  i, j, k: Integer;
begin
  // Do not zero the param grads -- thet is done in zero gradient.
  // As to the param values -- they are zeroed below.

  // Initialize embeddings.
  for i := 0 to nSymbols - 1 do             // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      WModelParams.Embeddings.Value[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  // Initialize per-block params.
  for k := 0 to nBlock - 1 do
    with WModelParams.ParamBlock[k] do begin

      // Initialize weight matrix W0.
      XGUniformW(W0.Value, ModelDim, ModelDim);

      // Initialize the weights with Xavier-Glorot function.
      XGUniformW(Wq.Value, ModelDim, ModelDim);
      XGUniformW(Wk.Value, ModelDim, ModelDim);
      XGUniformW(Wv.Value, ModelDim, ModelDim);

      // Initialize W1 and W2 weight matrices.
      XGUniformW1(W1.Value, ModelDim, ModelDimProj);
      XGUniformW2(W2.Value, ModelDimProj, ModelDim);

      // Initialize b1 and b2.
      FillChar(b1.Value, ProjectedSize, 0);
      FillChar(b2.Value, ModelSize, 0);

      // Initialize Beta and Gamma, LN 1 and 2, with SD and mean.
      FillChar(Beta1.Value, ModelSize, 0);
      FillChar(Beta2.Value, ModelSize, 0);
      for j := 0 to ModelDim - 1 do begin
        Gamma1.Value[j] := 1.0;
        Gamma2.Value[j] := 1.0;
      end;
    end;
end;

// Zero out all gradients.
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
begin
  with WModelState.StateBlock[Blk] do begin
    FillChar(Hidden1.Grad, HiddenSize, 0);
    FillChar(Hidden2.Grad, HiddenSize, 0);
    cudaMemset(X.dGrad, 0, XSize);
    cudaMemset(X1.dGrad, 0, XSize);
    cudaMemset(X2.dGrad, 0, XSize);
    cudaMemset(X3.dGrad, 0, XSize);
    cudaMemset(X4.dGrad, 0, XSize);
    cudaMemset(X5.dGrad, 0, XSize);
    cudaMemset(X6.dGrad, 0, XSize);
    cudaMemset(X7.dGrad, 0, XSize);
    cudaMemset(X1k.dGrad, 0, XSize);
    cudaMemset(X1q.dGrad, 0, XSize);
    cudaMemset(X1v.dGrad, 0, XSize);
    cudaMemset(K.dGrad, 0, XSize);
    cudaMemset(Q.dGrad, 0, XSize);
    cudaMemset(V.dGrad, 0, XSize);
    cudaMemset(Hidden1.dGrad, 0, HiddenSize);
    cudaMemset(Hidden2.dGrad, 0, HiddenSize);
  end;
  if Blk = 0 then
    cudaMemset(WModelParams.Embeddings.dGrad, 0, EmbeddingsSize);
  with WModelParams.ParamBlock[Blk] do begin
    cudaMemset(Wk.dGrad, 0, WeightSize);
    cudaMemset(Wq.dGrad, 0, WeightSize);
    cudaMemset(Wv.dGrad, 0, WeightSize);
    cudaMemset(W0.dGrad, 0, WeightSize);
    cudaMemset(W1.dGrad, 0, WeightProjectedSize);
    cudaMemset(W2.dGrad, 0, WeightProjectedSize);
    cudaMemset(b1.dGrad, 0, ProjectedSize);
    cudaMemset(b2.dGrad, 0, ModelSize);
    cudaMemset(Gamma1.dGrad, 0, ModelSize);
    cudaMemset(Gamma2.dGrad, 0, ModelSize);
    cudaMemset(Beta1.dGrad, 0, ModelSize);
    cudaMemset(Beta2.dGrad, 0, ModelSize);
  end;
end;

// Procedures for updating the parameters. Herw, with decay.
procedure CuUpdateParamDecay(Handle: TcublasHandle; const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
var
  Alpha: Single;
begin
  CuScale(Handle, N, DecayScale, Param);
  Alpha := -LearningRate;
  cublasSaxpy_v2(Handle, N, @Alpha, Grad, 1, Param, 1);
end;

// Procedures for updating the parameters. Herw, without decay.
procedure CuUpdateParamNoDecay(Handle: TcublasHandle; const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
var
  Alpha: Single;
begin
  Alpha := -LearningRate;
  cublasSaxpy_v2(Handle, N, @Alpha, Grad, 1, Param, 1);
end;

{Same as NoDecay
procedure CuUpdateParam(Handle: TcublasHandle; const N: Integer; const LearningRate: Single; const Grad: PSingle; Param: PSingle);
var
  Alpha: Single;
begin
  Alpha := -LearningRate;
  cublasSaxpy_v2(Handle, N, @Alpha, Grad, 1, Param, 1);
end;}

// Update the weights and biases. At some point, eliminate non-cublas updates.
procedure Optimization(var WModelParams: TWModelParams; const Blk: Integer);
begin
  with WModelParams.ParamBlock[Blk] do begin
    // W weights: main attention output.
    // UpdateParam(ModelDim * ModelDim, LearningRate, @W0.Grad[0,0], @W0.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDim * ModelDim, LearningRate, W0.dGrad, W0.dValue);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @W0.Grad[0, 0], 1, @W0.Value[0, 0], 1);

    // Wq, Wk, Wv weights: Q, K, V.
    // UpdateParam(ModelDim * ModelDim, LearningRate, @Wq.Grad[0,0], @Wq.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDim * ModelDim, LearningRate, Wq.dGrad, Wq.dValue);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wq.Grad[0, 0], 1, @Wq.Value[0, 0], 1);
    // UpdateParam(ModelDim * ModelDim, LearningRate, @Wk.Grad[0,0], @Wk.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDim * ModelDim, LearningRate, Wk.dGrad, Wk.dValue);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wk.Grad[0, 0], 1, @Wk.Value[0, 0], 1);
    // UpdateParam(ModelDim * ModelDim, LearningRate, @Wv.Grad[0,0], @Wv.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDim * ModelDim, LearningRate, Wv.dGrad, Wv.dValue);
    // cblas_saxpy(ModelDim * ModelDim, -LearningRate, @Wv.Grad[0, 0], 1, @Wv.Value[0, 0], 1);

    // UpdateParam(ModelDim * ModelDimProj, LearningRate, @W1.Grad[0,0], @W1.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDim * ModelDimProj, LearningRate, W1.dGrad, W1.dValue);
    // cblas_saxpy(ModelDim * ModelDimProj, -LearningRate, @W1.Grad[0, 0], 1, @W1.Value[0, 0], 1);
    // UpdateParam(ModelDimProj * ModelDim, LearningRate, @W2.Grad[0,0], @W2.Value[0,0]);
    CuUpdateParamDecay(CuHandle, ModelDimProj * ModelDim, LearningRate, W2.dGrad, W2.dValue);
    // cblas_saxpy(ModelDimProj * ModelDim, -LearningRate, @W2.Grad[0, 0], 1, @W2.Value[0, 0], 1);

    // b1, b2: biases.
    // UpdateParam(ModelDimProj, LearningRate, @b1.Grad[0], @b1.Value[0]);
    CuUpdateParamDecay(CuHandle, ModelDimProj, LearningRate, b1.dGrad, b1.dValue);
    // cblas_saxpy(ModelDimProj, -LearningRate, @b1.Grad[0], 1, @b1.Value[0], 1);
    // UpdateParam(ModelDim, LearningRate, @b2.Grad[0], @b2.Value[0]);
    CuUpdateParamDecay(CuHandle, ModelDim,     LearningRate, b2.dGrad, b2.dValue);
    // cblas_saxpy(ModelDim, -LearningRate, @b2.Grad[0], 1, @b2.Value[0], 1);

    // Gamma1, Gamm2, Beta1, Beta2: Layer-Norm parameters.
    // UpdateParam(ModelDim, LearningRate, @Gamma1.Grad[0], @Gamma1.Value[0]);
    CuUpdateParamNoDecay(CuHandle, ModelDim, LearningRate, Gamma1.dGrad, Gamma1.dValue);
    // cblas_saxpy(ModelDim, -LearningRate, @Gamma1.Grad[0], 1, @Gamma1.Value[0], 1);
    // UpdateParam(ModelDim, LearningRate, @Gamma2.Grad[0], @Gamma2.Value[0]);
    CuUpdateParamNoDecay(CuHandle, ModelDim, LearningRate, Gamma2.dGrad, Gamma2.dValue);
    // cblas_saxpy(ModelDim, -LearningRate, @Gamma2.Grad[0], 1, @Gamma2.Value[0], 1);
    // UpdateParam(ModelDim, LearningRate, @Beta1.Grad[0], @Beta1.Value[0]);
    CuUpdateParamNoDecay(CuHandle, ModelDim, LearningRate, Beta1.dGrad, Beta1.dValue);
    // cblas_saxpy(ModelDim, -LearningRate, @Beta1.Grad[0], 1, @Beta1.Value[0], 1);
    // UpdateParam(ModelDim, LearningRate, @Beta2.Grad[0], @Beta2.Value[0]);
    CuUpdateParamNoDecay(CuHandle, ModelDim, LearningRate, Beta2.dGrad, Beta2.dValue);
    // cblas_saxpy(ModelDim, -LearningRate, @Beta2.Grad[0], 1, @Beta2.Value[0], 1);

  end;
end;

// Update Embeddings.
procedure UpdateEmbeddings(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  // Add input-side embedding grads into Embeddings.dGrad.
  // Output-side tied gradient is already in Embeddings.dGrad.
  LaunchAddInputEmbeddingGrad(WModelState.StateBlock[0].X.dGrad, WModelParams.Embeddings.dGrad, dInputTokens, SeqLen, ModelDim, nVocab);

  // Apply total embedding gradient.
  CuUpdateParamNoDecay(CuHandle, nVocab * ModelDim, LearningRate, WModelParams.Embeddings.dGrad, WModelParams.Embeddings.dValue);
end;

// Rotary positional encoding. No longer used.
// Apply RoPE to both Q and K, [0..SeqLen - 1, 0..ModelDim - 1]
// Apply before head-splitting, immediately after computing Q and K.
procedure ApplyRoPE(var H: TSeqMatrix;  const InvFreq: TFVector; SeqLen, ModelDim: Integer);
var
  i, j: Integer;
  Angle, c, s, x0, x1: Single;
begin
  for i := 0 to SeqLen - 1 do
    for j := 0 to (ModelDim div 2) - 1 do begin
      Angle := i * InvFreq[j];
      c := Cos(Angle);
      s := Sin(Angle);

      // Original pair.
      x0 := H[i, 2 * j];
      x1 := H[i, 2 * j + 1];

      // Rotated pair.
      H[i, 2 * j]   :=  x0 * c - x1 * s;
      H[i, 2 * j + 1] :=  x0 * s + x1 * c;
    end;
end;

// Simple autoregressive masking. No longer used.
procedure ApplyAutoregressiveMask(var ScoresHead: TScoresMatrix; const L: Integer);
var
  i, j: Integer;
const
  NEG_INF: Single = -1e30;
begin
  for i := 0 to L - 1 do
    for j := i + 1 to L - 1 do
      ScoresHead[i, j] := NEG_INF;
end;

// Softmax ForwardN. No longer used.
procedure SoftmaxForwardN(const x: PSingle; y: PSingle; const N: Integer);
var
  i: Integer;
  MaxVal, SumVal, InvT: Single;
begin
  if N <= 0 then Exit;

  InvT := 1.0 / Temperature;

  // Find max for numerical stability.
  MaxVal := x[0] * InvT;
  for i := 1 to N - 1 do
    if (x[i] * InvT) > MaxVal then
      MaxVal := x[i] * InvT;

  // Exp and sum.
  SumVal := 0.0;
  for i := 0 to N - 1 do begin
    y[i] := Exp((x[i] * InvT) - MaxVal);
    SumVal := SumVal + y[i];
  end;

  // Normalize.
  if SumVal <> 0.0 then begin
    SumVal := 1.0 / SumVal;
    for i := 0 to N - 1 do
      y[i] := y[i] * SumVal;
  end;
end;

// Softmax procedure backward. No longer used.
procedure SoftmaxBackward(const y, dy:  TFVector; out dx: TFVector);
var
  j, D: Integer;
  dot: Single;
begin
  D := Length(y);
  SetLength(dx, D);

  // dot = sum_j dy[j] * y[j].
  dot := 0.0;
  for j := 0 to D - 1 do
    dot := dot + dy[j] * y[j];

  // dx[j] = y[j] * (dy[j] - dot).
  for j := 0 to D - 1 do
    dx[j] := y[j] * (dy[j] - dot);
end;

// Layer-Norm matrix. No longer used.
procedure LayerNormForward(const InX: TSeqMatrix; var OutX: TSeqMatrix; SeqLen: Integer;
  const Gamma, Beta: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  MeanL, VarL, InvStd: Single;
const
  EPS = 1e-5;
begin
  for i := 0 to SeqLen - 1 do begin
    MeanL := 0.0;
    for j := 0 to ModelDim - 1 do
      MeanL := MeanL + InX[i, j];
    MeanL := MeanL / ModelDim;

    VarL := 0.0;
    for j := 0 to ModelDim - 1 do
      VarL := VarL + Sqr(InX[i, j] - MeanL);
    VarL := VarL / ModelDim;

    InvStd := 1.0 / Sqrt(VarL + EPS);

    for j := 0 to ModelDim - 1 do
      OutX[i, j] := (InX[i, j] - MeanL) * InvStd * Gamma[j] + Beta[j];
    LNInvStd[i] := InvStd;
    for j := 0 to ModelDim - 1 do
      LNXhat[i, j] := (InX[i, j] - MeanL) * InvStd;
  end;
end;

// Layer-Norm matrix on back propagation. dY is upstream gradient. dX is output gradient.
// dGamma, dBeta are accumulated over all rows. No longer used.
procedure LayerNormBackward(const dY: TSeqMatrix; var dX: TSeqMatrix; var dGamma, dBeta: TSeqVector;
  SeqLen: Integer; const Gamma: TSeqVector; var LNXhat: TSeqMatrix; var LNInvStd: TFSVector);
var
  i, j: Integer;
  sum1, sum2, scale: Single;
  dHat: TSeqVector;
begin
  for i := 0 to SeqLen - 1 do begin
    // Step 1: dHat = dY * Gamma.
    sum1 := 0.0;
    sum2 := 0.0;
    for j := 0 to ModelDim - 1 do begin
      dHat[j] := dY[i, j] * Gamma[j];
      sum1 := sum1 + dHat[j];
      sum2 := sum2 + dHat[j] * LNXhat[i][j];
    end;

    // Step 2: compute dX.
    scale := LNInvStd[i] / ModelDim;
    for j := 0 to ModelDim - 1 do
      dX[i, j] := scale * (ModelDim * dHat[j] - sum1 - LNXhat[i, j] * sum2);

    // Step 3: accumulate dGamma and dBeta.
    for j := 0 to ModelDim - 1 do begin
      dGamma[j] := dGamma[j] + dY[i, j] * LNXhat[i][j];
      dBeta[j]  := dBeta[j]  + dY[i, j];
    end;
  end;
end;

// Calculate cross-entropy gradient from probabilities and target, one-hot. No longer used.
procedure GradientFromCEProbabilities(var WModelState: TWModelState);
var
  i, s: Integer;
begin
  with WModelState do
    for i := 0 to SeqLen - 1 do begin
      for s := 0 to nVocab - 1 do
        TopGradient[i, s] := Probs[i, s];

      TopGradient[i, TargetTokens[i]] :=
        Probs[i, TargetTokens[i]] - 1.0;
    end;
end;

// Calculate gradient for KL divergence with one-hot targets: dL/dProbs = Q - P. No longer used.
procedure GradientFromKLDivergence(var WModelState: TWModelState);
var
  i, s: Integer;
begin
  with WModelState do
    for i := 0 to SeqLen - 1 do
      for s := 0 to nVocab - 1 do
        if s = TargetTokens[i] then
          TopGradient[i, s] := Probs[i, s] - 1.0   // Q - 1.
        else
          TopGradient[i, s] := Probs[i, s] - 0.0;  // Q - 0.
end;

// Back propagation addition. No longer used.
procedure BackpropAdd(const dOut: TSeqMatrix; var dA, dB: TSeqMatrix; const L, D: Integer);
var
  i, j: Integer;
begin
  for i := 0 to L - 1 do
    for j := 0 to D - 1 do begin
      dA[i, j] := dA[i, j] + dOut[i, j];
      dB[i, j] := dB[i, j] + dOut[i, j];
    end;
end;

end.

