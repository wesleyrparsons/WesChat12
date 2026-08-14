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
  InvFreqSize: Integer = (HeadDim div 2) * SizeOf(Single);
  ProbsSize: Integer = SeqLen * DimVocab * SizeOf(Single);

// Cublas and Cuda procedures.
procedure InitializeCublas;
procedure CheckCudaError(const Where: string);
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);

// Saving routines.
function IsAbsolutePath(const FolderName: string): Boolean;
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

// Loss routines.
function ComputeLoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;

// Initialization routines.
procedure XGUniformW(var W: TWeightMatrix; FanIn, FanOut: Integer);
procedure XGUniformW1(var W: TWeightProjMatrix; FanIn, FanOut: Integer);
procedure XGUniformW2(var W: TWeightProjMatrixT; FanIn, FanOut: Integer);
procedure InitializeTransformerState(var WModelState: TWModelState);
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
procedure CopyParamsToDevice(var WModelParams: TWModelParams);
procedure CopyParamsToHost(var WModelParams: TWModelParams);
procedure CopyAdamWStateToHost(var WAdamWState: TWAdamWState);
procedure CopyAdamWStateToDevice(var WAdamWState: TWAdamWState);
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
procedure InitializeWAdamWState(var WAdamWState: TWAdamWState);
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);

// Optimization routines.
procedure AdamWOptimizeBlock(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Blk: Integer;
  const Beta1Power, Beta2Power: Single);
procedure AdamWOptimizeEmbeddings(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Beta1Power, Beta2Power: Single);
procedure UpdateEmbeddingGradient(var WModelParams: TWModelParams; var WModelState: TWModelState);
procedure GetAdamWStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState;
  out ParamRMS, UpdateRMS, UpdateRatio, MRMS, SqrtVRMS: Double);

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
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchSoftmaxForwardStrided(dIn: PSingle; dOut: PSingle; Rows: Integer; Cols: Integer; RowStride: Integer; Temperature: Single);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchSoftmaxForward(dIn: PSingle; dOut: PSingle; Rows: Integer; N: Integer; Temperature: Single);
procedure LaunchSoftmaxBackward(Y: PSingle; dY: PSingle; dX: PSingle; Rows: Integer; D: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchLayerNormForward(InX, OutX, Gamma, Beta, LNXhat, LNInvStd: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchLayerNormBackward(dY, dX, Gamma, LNXhat, LNInvStd, dGamma, dBeta: PSingle; SeqLen, ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchRoPEForward(H: PSingle; const InvFreq: PSingle; SeqLen: Integer; NumHeads: Integer; HeadDim: Integer; RowStride: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchRoPEBackward(dH: PSingle; const InvFreq: PSingle; SeqLen: Integer; NumHeads: Integer; HeadDim: Integer; RowStride: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchRoPEBackward(dH: PSingle; InvFreq: PSingle; SeqLen: Integer; ModelDim: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchCEGradient(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; SeqLen: Integer; nVocab: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchCEGradientStrided(Probs: PSingle; TopGradient: PSingle; TargetTokens: PInteger; Rows: Integer; VocabSize: Integer; RowStride: Integer; GradScale: Single);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddInputEmbeddingGrad(XGrad: PSingle; EmbGrad: PSingle; InputTokens: PInteger; SeqLen: Integer; ModelDim: Integer; nVocab: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddBiasRows(X: PSingle; Bias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAddBiasRowsBackward(dX: PSingle; dBias: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchCELossRows(Probs: PSingle; Targets: PInteger; RowLoss: PSingle; Rows, VocabSize, RowStride: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchAdamWUpdate(Param, Grad, M, V: PSingle; Count: Integer; LearningRate, Beta1, Beta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay: Single);
  cdecl; external 'WesChatKernel12.dll';

implementation

{ DLL procedures }
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

{ Cublas and cuda procedures }
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
  Err := cudaGetLastError;
  if Err <> 0 then begin
    Writeln;
    Writeln('CUDA LAUNCH ERROR. Location: ', Where, '; Error # : ', Err, '; Message : ', StrPas(cudaGetErrorString(Err)), '.');
    Pause;
    Exit;
  end;

  Err := cudaDeviceSynchronize;
  if Err <> 0 then begin
    Writeln;
    Writeln('CUDA EXECUTION ERROR. Location: ', Where, '; Error # : ', Err, '; Message : ', StrPas(cudaGetErrorString(Err)), '.');
    Pause;
  end;
end;

// Check cuda status.
procedure CheckCudaStatus(const Status: Integer; const Where: string);
begin
  if Status <> 0 then begin
    Writeln('CUDA error in ', Where, ': ', StrPas(cudaGetErrorString(Status)), '.');
    Pause;
    Halt;
  end;
end;

// Intialize Cuda and Cublas.
procedure StartCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
begin
  InitializeCublas;
  if not CudaAllocated then
    MAllocCublas(WModelParams, WModelState, WAdamWState);
  CheckCudaError('Cuda started.');
end;

// End Cuda and Cublas.
procedure EndCuda(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
begin
  if CudaAllocated then
    MDeallocateCublas(WModelParams, WModelState, WAdamWState);
  if CuBLAS_Shutdown then
    Writeln('CuBLAS successfully shut down.')
end;

{ Reports }
// Full compute of trainable parameters.
procedure FullReportTrainableParameters;
var
  EmbeddingParams, AttentionParams, FFNParams, LayerNormParams, BlockParams, TotalParams: Int64;
begin
  EmbeddingParams := Int64(nVocab) * ModelDim;

  // Wq, Wk, Wv, W0.
  AttentionParams := Int64(4) * ModelDim * ModelDim;

  // W1, W2, b1, b2.
  FFNParams := Int64(2) * ModelDim * ModelDimProj + ModelDimProj + ModelDim;

  // Gamma1, Beta1, Gamma2, Beta2.
  LayerNormParams := Int64(4) * ModelDim;

  BlockParams := AttentionParams + FFNParams + LayerNormParams;
  TotalParams := EmbeddingParams + Int64(nBlock) * BlockParams;

  Writeln('Trainable parameters:');
  Writeln('  Embeddings       = ', EmbeddingParams);
  Writeln('  Attention/block  = ', AttentionParams);
  Writeln('  FFN/block        = ', FFNParams);
  Writeln('  LayerNorm/block  = ', LayerNormParams);
  Writeln('  Total/block      = ', BlockParams);
  Writeln('  Blocks           = ', nBlock);
  Writeln('  Total parameters = ', TotalParams);
end;

// Short report of trainable parameters.
function NumberTrainableParameters: Int64;
var
  EmbeddingParams, AttentionParams, FFNParams, LayerNormParams, BlockParams, TotalParams: Int64;
begin
  EmbeddingParams := Int64(nVocab) * ModelDim;

  // Wq, Wk, Wv, W0.
  AttentionParams := Int64(4) * ModelDim * ModelDim;

  // W1, W2, b1, b2.
  FFNParams := Int64(2) * ModelDim * ModelDimProj + ModelDimProj + ModelDim;

  // Gamma1, Beta1, Gamma2, Beta2.
  LayerNormParams := Int64(4) * ModelDim;

  BlockParams := AttentionParams + FFNParams + LayerNormParams;
  TotalParams := EmbeddingParams + Int64(nBlock) * BlockParams;

  Result := TotalParams;
end;

{ Saving routines.}
// Absolute path.
function IsAbsolutePath(const FolderName: string): Boolean;
begin
  Result := (ExtractFileDrive(FolderName) <> '') or
    ((Length(FolderName) >= 2) and (FolderName[1] = DirectorySeparator) and (FolderName[2] = DirectorySeparator));
end;

// Initialize work folders.
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

// Clean base name.
function CleanBaseName(const FileName: string): string;
begin
  Result := ChangeFileExt(ExtractFileName(FileName), '');
end;

// Create work symbol file name.
function WorkSymbolFile(const BaseName: string): string;
begin
  Result := SymbolDir + ChangeFileExt(ExtractFileName(BaseName), '.sym.tok');
end;

// Create work token file name.
function WorkTokenFile(const BaseName: string): string;
begin
  Result := TokenDir + ChangeFileExt(ExtractFileName(BaseName), '.tok');
end;

// Create work model file name.
function WorkModelFile(const BaseName: string): string;
begin
  Result := ModelDir + ChangeFileExt(ExtractFileName(BaseName), '.model');
end;

// Create work log file name.
function WorkLogFile(const BaseName: string): string;
begin
  Result := LogDir + ChangeFileExt(ExtractFileName(BaseName), '.log');
end;

// Create work run file name.
function WorkRunFile(const BaseName: string): string;
begin
  Result := RunDir + BaseName + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.run';
end;

{ Utility procedures }
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

  Write('Detokenized Corpus (in full length of ', Length(TC), '): ');
  for i := 0 to High(TC) do
    Write(Decode(TC[i]));
  Writeln;
  Pause;
end;

// Decode for WesTokenize and GPT2Tokenize, using symbol table, for one token.
function Decode(const x: Integer): UnicodeString;
begin
  if Tokenizer = WesTokenizer then begin
    Result := UTF8Decode(SymbolTable[x]);
    Exit;
  end;

  if x = GPT2BOS then
    Result := '<BOS>'
  else if x = GPT2EOS then
    Result := '<EOS>'
  else if x = GPT2PAD then
    Result := '<PAD>'
  else if x = GPT2UNK then
    Result := '<UNK>'
  else
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

{ Initialization routines }
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

// Allocate CUDA memory for model parameters, model state, and AdamW state.
procedure MAllocCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
var
  h, k: Integer;

  procedure AllocSingle(var P: PSingle; const Bytes: Integer; const Name: string);
  begin
    CheckCudaStatus(cudaMalloc(@P, Bytes), Name);
  end;

  procedure AllocInteger(var P: PInteger; const Bytes: Integer; const Name: string);
  begin
    CheckCudaStatus(cudaMalloc(@P, Bytes), Name);
  end;

begin
  CudaAllocated := False;

  // Input and target tokens.
  AllocInteger(dInputTokens, SeqLen * SizeOf(Integer), 'dInputTokens');
  AllocInteger(dTargetTokens, SeqLen * SizeOf(Integer), 'dTargetTokens');

  // Global/shared model parameters.
  with WModelParams do begin
    AllocSingle(Embeddings.dValue, EmbeddingsSize, 'Embeddings.dValue');
    AllocSingle(Embeddings.dGrad, EmbeddingsSize, 'Embeddings.dGrad');
  end;

  // Global/shared AdamW state.
  with WAdamWState.Embeddings do begin
    AllocSingle(dM, EmbeddingsSize, 'Adam Embeddings.dM');
    AllocSingle(dV, EmbeddingsSize, 'Adam Embeddings.dV');
  end;

  // Global/shared model state.
  with WModelState do begin
    AllocSingle(dInvFreq, InvFreqSize, 'dInvFreq');
    AllocSingle(dProbs, ProbsSize, 'dProbs');
    AllocSingle(dRowLoss, SeqSize, 'dRowLoss');
    AllocSingle(dTopGradient, ProbsSize, 'dTopGradient');
  end;

  // Per-block parameters, AdamW state, and transformer state.
  for k := 0 to nBlock - 1 do begin

    // Trainable model parameters.
    with WModelParams.ParamBlock[k] do begin
      AllocSingle(Wq.dValue, WeightSize, 'Wq.dValue');
      AllocSingle(Wq.dGrad, WeightSize, 'Wq.dGrad');

      AllocSingle(Wk.dValue, WeightSize, 'Wk.dValue');
      AllocSingle(Wk.dGrad, WeightSize, 'Wk.dGrad');

      AllocSingle(Wv.dValue, WeightSize, 'Wv.dValue');
      AllocSingle(Wv.dGrad, WeightSize, 'Wv.dGrad');

      AllocSingle(W0.dValue, WeightSize, 'W0.dValue');
      AllocSingle(W0.dGrad, WeightSize, 'W0.dGrad');

      AllocSingle(W1.dValue, WeightProjectedSize, 'W1.dValue');
      AllocSingle(W1.dGrad, WeightProjectedSize, 'W1.dGrad');

      AllocSingle(W2.dValue, WeightProjectedSize, 'W2.dValue');
      AllocSingle(W2.dGrad, WeightProjectedSize, 'W2.dGrad');

      AllocSingle(b1.dValue, ProjectedSize, 'b1.dValue');
      AllocSingle(b1.dGrad, ProjectedSize, 'b1.dGrad');

      AllocSingle(b2.dValue, ModelSize, 'b2.dValue');
      AllocSingle(b2.dGrad, ModelSize, 'b2.dGrad');

      AllocSingle(Gamma1.dValue, ModelSize, 'Gamma1.dValue');
      AllocSingle(Gamma1.dGrad, ModelSize, 'Gamma1.dGrad');

      AllocSingle(Beta1.dValue, ModelSize, 'Beta1.dValue');
      AllocSingle(Beta1.dGrad, ModelSize, 'Beta1.dGrad');

      AllocSingle(Gamma2.dValue, ModelSize, 'Gamma2.dValue');
      AllocSingle(Gamma2.dGrad, ModelSize, 'Gamma2.dGrad');

      AllocSingle(Beta2.dValue, ModelSize, 'Beta2.dValue');
      AllocSingle(Beta2.dGrad, ModelSize, 'Beta2.dGrad');
    end;

    // Persistent AdamW first and second moments.
    with WAdamWState.ParamBlock[k] do begin
      AllocSingle(Wq.dM, WeightSize, 'Adam Wq.dM');
      AllocSingle(Wq.dV, WeightSize, 'Adam Wq.dV');

      AllocSingle(Wk.dM, WeightSize, 'Adam Wk.dM');
      AllocSingle(Wk.dV, WeightSize, 'Adam Wk.dV');

      AllocSingle(Wv.dM, WeightSize, 'Adam Wv.dM');
      AllocSingle(Wv.dV, WeightSize, 'Adam Wv.dV');

      AllocSingle(W0.dM, WeightSize, 'Adam W0.dM');
      AllocSingle(W0.dV, WeightSize, 'Adam W0.dV');

      AllocSingle(W1.dM, WeightProjectedSize, 'Adam W1.dM');
      AllocSingle(W1.dV, WeightProjectedSize, 'Adam W1.dV');

      AllocSingle(W2.dM, WeightProjectedSize, 'Adam W2.dM');
      AllocSingle(W2.dV, WeightProjectedSize, 'Adam W2.dV');

      AllocSingle(b1.dM, ProjectedSize, 'Adam b1.dM');
      AllocSingle(b1.dV, ProjectedSize, 'Adam b1.dV');

      AllocSingle(b2.dM, ModelSize, 'Adam b2.dM');
      AllocSingle(b2.dV, ModelSize, 'Adam b2.dV');

      AllocSingle(Gamma1.dM, ModelSize, 'Adam Gamma1.dM');
      AllocSingle(Gamma1.dV, ModelSize, 'Adam Gamma1.dV');

      AllocSingle(Beta1.dM, ModelSize, 'Adam Beta1.dM');
      AllocSingle(Beta1.dV, ModelSize, 'Adam Beta1.dV');

      AllocSingle(Gamma2.dM, ModelSize, 'Adam Gamma2.dM');
      AllocSingle(Gamma2.dV, ModelSize, 'Adam Gamma2.dV');

      AllocSingle(Beta2.dM, ModelSize, 'Adam Beta2.dM');
      AllocSingle(Beta2.dV, ModelSize, 'Adam Beta2.dV');
    end;

    // Non-trainable transformer state.
    with WModelState.StateBlock[k] do begin
      AllocSingle(X.dValue, XSize, 'X.dValue');
      AllocSingle(X.dGrad, XSize, 'X.dGrad');

      AllocSingle(X1.dValue, XSize, 'X1.dValue');
      AllocSingle(X1.dGrad, XSize, 'X1.dGrad');

      AllocSingle(X2.dValue, XSize, 'X2.dValue');
      AllocSingle(X2.dGrad, XSize, 'X2.dGrad');

      AllocSingle(X3.dValue, XSize, 'X3.dValue');
      AllocSingle(X3.dGrad, XSize, 'X3.dGrad');

      AllocSingle(X4.dValue, XSize, 'X4.dValue');
      AllocSingle(X4.dGrad, XSize, 'X4.dGrad');

      AllocSingle(X5.dValue, XSize, 'X5.dValue');
      AllocSingle(X5.dGrad, XSize, 'X5.dGrad');

      AllocSingle(X6.dValue, XSize, 'X6.dValue');
      AllocSingle(X6.dGrad, XSize, 'X6.dGrad');

      AllocSingle(X7.dValue, XSize, 'X7.dValue');
      AllocSingle(X7.dGrad, XSize, 'X7.dGrad');

      AllocSingle(X1q.dValue, XSize, 'X1q.dValue');
      AllocSingle(X1q.dGrad, XSize, 'X1q.dGrad');

      AllocSingle(X1k.dValue, XSize, 'X1k.dValue');
      AllocSingle(X1k.dGrad, XSize, 'X1k.dGrad');

      AllocSingle(X1v.dValue, XSize, 'X1v.dValue');
      AllocSingle(X1v.dGrad, XSize, 'X1v.dGrad');

      AllocSingle(Q.dValue, XSize, 'Q.dValue');
      AllocSingle(Q.dGrad, XSize, 'Q.dGrad');

      AllocSingle(K.dValue, XSize, 'K.dValue');
      AllocSingle(K.dGrad, XSize, 'K.dGrad');

      AllocSingle(V.dValue, XSize, 'V.dValue');
      AllocSingle(V.dGrad, XSize, 'V.dGrad');

      for h := 0 to nHead - 1 do begin
        AllocSingle(ScoresHead1[h].dValue, ScoresSize, 'ScoresHead1.dValue');
        AllocSingle(ScoresHead1[h].dGrad, ScoresSize, 'ScoresHead1.dGrad');
        AllocSingle(ScoresHead2[h].dValue, ScoresSize, 'ScoresHead2.dValue');
        AllocSingle(ScoresHead2[h].dGrad, ScoresSize, 'ScoresHead2.dGrad');
      end;

      AllocSingle(Hidden1.dValue, HiddenSize, 'Hidden1.dValue');
      AllocSingle(Hidden1.dGrad, HiddenSize, 'Hidden1.dGrad');

      AllocSingle(Hidden2.dValue, HiddenSize, 'Hidden2.dValue');
      AllocSingle(Hidden2.dGrad, HiddenSize, 'Hidden2.dGrad');

      AllocSingle(dLNInvStd1, SeqSize, 'dLNInvStd1');
      AllocSingle(dLNXHat1, XSize, 'dLNXHat1');

      AllocSingle(dLNInvStd2, SeqSize, 'dLNInvStd2');
      AllocSingle(dLNXHat2, XSize, 'dLNXHat2');

      AllocSingle(dX4FromLN2, XSize, 'dX4FromLN2');
      AllocSingle(dXFromLN1, XSize, 'dXFromLN1');
    end;
  end;

  CheckCudaError('Allocate CUDA memory.');
  CudaAllocated := True;
end;

// De-allocate cublas memory.
procedure MDeallocateCublas(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState);
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
    if dRowLoss <> nil then begin
      cudaFree(dRowLoss);
      dRowLoss := nil;
    end;
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
  ParamsNeedCopyToDevice := True;
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
  ParamsNeedCopyToDevice := False;
  if DebugCudaChecks then
    CheckCudaError('Copy model parameters to device.');
end;

// Copy AdamW first and second moments from host to CUDA device.
procedure CopyAdamWStateToDevice(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemcpy(dM, @M[0, 0], EmbeddingsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, @V[0, 0], EmbeddingsSize, cudaMemcpyHostToDevice);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemcpy(Wq.dM, @Wq.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wq.dV, @Wq.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Wk.dM, @Wk.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wk.dV, @Wk.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Wv.dM, @Wv.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Wv.dV, @Wv.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      cudaMemcpy(W0.dM, @W0.M[0, 0], WeightSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W0.dV, @W0.V[0, 0], WeightSize, cudaMemcpyHostToDevice);

      // MLP weights.
      cudaMemcpy(W1.dM, @W1.M[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W1.dV, @W1.V[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);

      cudaMemcpy(W2.dM, @W2.M[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(W2.dV, @W2.V[0, 0], WeightProjectedSize, cudaMemcpyHostToDevice);

      // Biases.
      cudaMemcpy(b1.dM, @b1.M[0], ProjectedSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b1.dV, @b1.V[0], ProjectedSize, cudaMemcpyHostToDevice);

      cudaMemcpy(b2.dM, @b2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(b2.dV, @b2.V[0], ModelSize, cudaMemcpyHostToDevice);

      // LayerNorm parameters.
      cudaMemcpy(Gamma1.dM, @Gamma1.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma1.dV, @Gamma1.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Beta1.dM, @Beta1.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta1.dV, @Beta1.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Gamma2.dM, @Gamma2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Gamma2.dV, @Gamma2.V[0], ModelSize, cudaMemcpyHostToDevice);

      cudaMemcpy(Beta2.dM, @Beta2.M[0], ModelSize, cudaMemcpyHostToDevice);
      cudaMemcpy(Beta2.dV, @Beta2.V[0], ModelSize, cudaMemcpyHostToDevice);
    end;

  if DebugCudaChecks then
    CheckCudaError('Copy AdamW state to device.');
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
  if DebugCudaChecks then
    CheckCudaError('Copy model parameters to host.');
end;

// Copy AdamW first and second moments from CUDA device to host.
procedure CopyAdamWStateToHost(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemcpy(@M[0, 0], dM, EmbeddingsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(@V[0, 0], dV, EmbeddingsSize, cudaMemcpyDeviceToHost);
  end;

  // Per-block AdamW state.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemcpy(@Wq.M[0, 0], Wq.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wq.V[0, 0], Wq.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Wk.M[0, 0], Wk.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wk.V[0, 0], Wk.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Wv.M[0, 0], Wv.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Wv.V[0, 0], Wv.dV, WeightSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@W0.M[0, 0], W0.dM, WeightSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W0.V[0, 0], W0.dV, WeightSize, cudaMemcpyDeviceToHost);

      // MLP weights.
      cudaMemcpy(@W1.M[0, 0], W1.dM, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W1.V[0, 0], W1.dV, WeightProjectedSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@W2.M[0, 0], W2.dM, WeightProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@W2.V[0, 0], W2.dV, WeightProjectedSize, cudaMemcpyDeviceToHost);

      // Biases.
      cudaMemcpy(@b1.M[0], b1.dM, ProjectedSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b1.V[0], b1.dV, ProjectedSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@b2.M[0], b2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@b2.V[0], b2.dV, ModelSize, cudaMemcpyDeviceToHost);

      // LayerNorm parameters.
      cudaMemcpy(@Gamma1.M[0], Gamma1.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma1.V[0], Gamma1.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Beta1.M[0], Beta1.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta1.V[0], Beta1.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Gamma2.M[0], Gamma2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Gamma2.V[0], Gamma2.dV, ModelSize, cudaMemcpyDeviceToHost);

      cudaMemcpy(@Beta2.M[0], Beta2.dM, ModelSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(@Beta2.V[0], Beta2.dV, ModelSize, cudaMemcpyDeviceToHost);
    end;

  if DebugCudaChecks then
    CheckCudaError('Copy AdamW state to host.');
end;

// Copy InvFreq from host to device.
procedure CopyInvFreqToDevice(var WModelState: TWModelState);
begin
  cudaMemcpy(WModelState.dInvFreq, @WModelState.InvFreq[0], InvFreqSize, cudaMemcpyHostToDevice);
  if DebugCudaChecks then
    CheckCudaError('Copy inverse frequencies to device.');
end;

// Initialize the transformer state.
procedure InitializeTransformerState(var WModelState: TWModelState);
var
  j: Integer;
begin
  // Do not zero the state grads -- thet is done in zero gradient.
  // Do not zero the state values -- that is not necessary.

  // Compute InvFreq.
  SetLength(WModelState.InvFreq, HeadDim div 2);

  for j := 0 to (HeadDim div 2) - 1 do
    WModelState.InvFreq[j] := Exp(-(2.0 * j) / HeadDim * Ln(10000.0));
end;

// Initialize the transformer state stage.
procedure InitializeTransformerParams(var WModelParams: TWModelParams);
var
  i, j, k: Integer;
begin
  // Do not zero the param grads -- that is done in zero gradient.
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

// Initialize AdamW first and second moments to zero.
// Call only when starting a new AdamW optimizer.
procedure InitializeWAdamWState(var WAdamWState: TWAdamWState);
var
  k: Integer;
begin
  // Tied embeddings.
  with WAdamWState.Embeddings do begin
    cudaMemset(dM, 0, EmbeddingsSize);
    cudaMemset(dV, 0, EmbeddingsSize);
  end;

  // Per-block AdamW moments.
  for k := 0 to nBlock - 1 do
    with WAdamWState.ParamBlock[k] do begin

      // Attention weights.
      cudaMemset(Wq.dM, 0, WeightSize);
      cudaMemset(Wq.dV, 0, WeightSize);

      cudaMemset(Wk.dM, 0, WeightSize);
      cudaMemset(Wk.dV, 0, WeightSize);

      cudaMemset(Wv.dM, 0, WeightSize);
      cudaMemset(Wv.dV, 0, WeightSize);

      cudaMemset(W0.dM, 0, WeightSize);
      cudaMemset(W0.dV, 0, WeightSize);

      // MLP weights.
      cudaMemset(W1.dM, 0, WeightProjectedSize);
      cudaMemset(W1.dV, 0, WeightProjectedSize);

      cudaMemset(W2.dM, 0, WeightProjectedSize);
      cudaMemset(W2.dV, 0, WeightProjectedSize);

      // Biases.
      cudaMemset(b1.dM, 0, ProjectedSize);
      cudaMemset(b1.dV, 0, ProjectedSize);

      cudaMemset(b2.dM, 0, ModelSize);
      cudaMemset(b2.dV, 0, ModelSize);

      // LayerNorm parameters.
      cudaMemset(Gamma1.dM, 0, ModelSize);
      cudaMemset(Gamma1.dV, 0, ModelSize);

      cudaMemset(Beta1.dM, 0, ModelSize);
      cudaMemset(Beta1.dV, 0, ModelSize);

      cudaMemset(Gamma2.dM, 0, ModelSize);
      cudaMemset(Gamma2.dV, 0, ModelSize);

      cudaMemset(Beta2.dM, 0, ModelSize);
      cudaMemset(Beta2.dV, 0, ModelSize);
    end;

  if DebugCudaChecks then
    CheckCudaError('Initialize AdamW state.');
end;

// Zero out all gradients.
procedure ZeroGradients(var WModelParams: TWModelParams; var WModelState: TWModelState; const Blk: Integer);
var
  h: Integer;
begin
  with WModelState.StateBlock[Blk] do begin
    // Optional host-side diagnostic buffers.
    FillChar(Hidden1.Grad, HiddenSize, 0);
    FillChar(Hidden2.Grad, HiddenSize, 0);

    // SeqLen x ModelDim state gradients.
    cudaMemset(X.dGrad,   0, XSize);
    cudaMemset(X1.dGrad,  0, XSize);
    cudaMemset(X2.dGrad,  0, XSize);
    cudaMemset(X3.dGrad,  0, XSize);
    cudaMemset(X4.dGrad,  0, XSize);
    cudaMemset(X5.dGrad,  0, XSize);
    cudaMemset(X6.dGrad,  0, XSize);
    cudaMemset(X7.dGrad,  0, XSize);

    cudaMemset(Q.dGrad,   0, XSize);
    cudaMemset(K.dGrad,   0, XSize);
    cudaMemset(V.dGrad,   0, XSize);

    cudaMemset(X1q.dGrad, 0, XSize);
    cudaMemset(X1k.dGrad, 0, XSize);
    cudaMemset(X1v.dGrad, 0, XSize);

    // SeqLen x ModelDimProj state gradients.
    cudaMemset(Hidden1.dGrad, 0, HiddenSize);
    cudaMemset(Hidden2.dGrad, 0, HiddenSize);

    // SeqLen x SeqLen attention gradients for each head.
    for h := 0 to nHead - 1 do begin
      cudaMemset(ScoresHead1[h].dGrad, 0, ScoresSize);
      cudaMemset(ScoresHead2[h].dGrad, 0, ScoresSize);
    end;
  end;

  // Embeddings are shared by all blocks, so clear them once.
  if Blk = 0 then
    cudaMemset(WModelParams.Embeddings.dGrad, 0, EmbeddingsSize);

  with WModelParams.ParamBlock[Blk] do begin
    // Attention parameters.
    cudaMemset(Wq.dGrad, 0, WeightSize);
    cudaMemset(Wk.dGrad, 0, WeightSize);
    cudaMemset(Wv.dGrad, 0, WeightSize);
    cudaMemset(W0.dGrad, 0, WeightSize);

    // FFN parameters.
    cudaMemset(W1.dGrad, 0, WeightProjectedSize);
    cudaMemset(W2.dGrad, 0, WeightProjectedSize);
    cudaMemset(b1.dGrad, 0, ProjectedSize);
    cudaMemset(b2.dGrad, 0, ModelSize);

    // Layer-normalization parameters.
    cudaMemset(Gamma1.dGrad, 0, ModelSize);
    cudaMemset(Beta1.dGrad,  0, ModelSize);
    cudaMemset(Gamma2.dGrad, 0, ModelSize);
    cudaMemset(Beta2.dGrad,  0, ModelSize);
  end;
end;

{ Optimization routines }
// Update one transformer block using AdamW.
procedure AdamWOptimizeBlock(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Blk: Integer;
  const Beta1Power, Beta2Power: Single);
begin
  with WAdamWState.ParamBlock[Blk] do
  with WModelParams.ParamBlock[Blk] do begin

    // Attention weights. Apply weight decay.
    LaunchAdamWUpdate(Wq.dValue, Wq.dGrad, WAdamWState.ParamBlock[Blk].Wq.dM, WAdamWState.ParamBlock[Blk].Wq.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(Wk.dValue, Wk.dGrad, WAdamWState.ParamBlock[Blk].Wk.dM, WAdamWState.ParamBlock[Blk].Wk.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(Wv.dValue, Wv.dGrad, WAdamWState.ParamBlock[Blk].Wv.dM, WAdamWState.ParamBlock[Blk].Wv.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(W0.dValue, W0.dGrad, WAdamWState.ParamBlock[Blk].W0.dM, WAdamWState.ParamBlock[Blk].W0.dV,
      ModelDim * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    // MLP weights. Apply weight decay.
    LaunchAdamWUpdate(W1.dValue, W1.dGrad, WAdamWState.ParamBlock[Blk].W1.dM, WAdamWState.ParamBlock[Blk].W1.dV,
      ModelDim * ModelDimProj, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    LaunchAdamWUpdate(W2.dValue, W2.dGrad, WAdamWState.ParamBlock[Blk].W2.dM, WAdamWState.ParamBlock[Blk].W2.dV,
      ModelDimProj * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

    // Biases. No weight decay.
    LaunchAdamWUpdate(b1.dValue, b1.dGrad, WAdamWState.ParamBlock[Blk].b1.dM, WAdamWState.ParamBlock[Blk].b1.dV,
      ModelDimProj, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    LaunchAdamWUpdate(b2.dValue, b2.dGrad, WAdamWState.ParamBlock[Blk].b2.dM, WAdamWState.ParamBlock[Blk].b2.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // LayerNorm 1. No weight decay.
    LaunchAdamWUpdate(Gamma1.dValue, Gamma1.dGrad, WAdamWState.ParamBlock[Blk].Gamma1.dM, WAdamWState.ParamBlock[Blk].Gamma1.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    LaunchAdamWUpdate(Beta1.dValue, Beta1.dGrad, WAdamWState.ParamBlock[Blk].Beta1.dM, WAdamWState.ParamBlock[Blk].Beta1.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    // LayerNorm 2. No weight decay.
    LaunchAdamWUpdate(Gamma2.dValue, Gamma2.dGrad, WAdamWState.ParamBlock[Blk].Gamma2.dM, WAdamWState.ParamBlock[Blk].Gamma2.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);

    LaunchAdamWUpdate(Beta2.dValue, Beta2.dGrad, WAdamWState.ParamBlock[Blk].Beta2.dM, WAdamWState.ParamBlock[Blk].Beta2.dV,
      ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, 0.0);
  end;

  if DebugCudaChecks then
    CheckCudaError('AdamW optimizer block ' + IntToStr(Blk));
end;

// AdamW optimize embeddings.
procedure AdamWOptimizeEmbeddings(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState; const Beta1Power, Beta2Power: Single);
begin
  with WModelParams.Embeddings do
    LaunchAdamWUpdate(dValue, dGrad, WAdamWState.Embeddings.dM, WAdamWState.Embeddings.dV,
      nVocab * ModelDim, LearningRate, AdamBeta1, AdamBeta2, Beta1Power, Beta2Power, AdamEpsilon, WeightDecay);

  if DebugCudaChecks then
    CheckCudaError('AdamW optimizer embeddings.');
end;

// Accumulate AdamW statistics for one parameter tensor. Param, M, and V are HOST pointers.
procedure AccumulateAdamWStatistics(Param, M, V: PSingle; const Count: Integer; const Decay: Single;
  const Beta1Power, Beta2Power: Double; var SumParam2, SumUpdate2, SumM2, SumV: Double; var TotalCount: Int64);
var
  i: Integer;
  PNew, POld, MV, VV, MHat, VHat, AdamTerm, Update, Denom: Double;
begin
  Denom := 1.0 - LearningRate * Decay;

  for i := 0 to Count - 1 do begin
    PNew := Param^;
    MV := M^;
    VV := V^;

    // Bias-corrected AdamW moments for the most recent update.
    MHat := MV / (1.0 - Beta1Power);
    VHat := VV / (1.0 - Beta2Power);

    if VHat < 0.0 then
      VHat := 0.0;

    AdamTerm := MHat / (Sqrt(VHat) + AdamEpsilon);

    // AdamW kernel did: PNew = POld - LR * (AdamTerm + Decay * POld). Therefore reconstruct POld.
    POld := (PNew + LearningRate * AdamTerm) / Denom;

    Update := PNew - POld;

    SumParam2 := SumParam2 + POld * POld;
    SumUpdate2 := SumUpdate2 + Update * Update;
    SumM2 := SumM2 + MV * MV;
    SumV := SumV + VV;

    Inc(Param);
    Inc(M);
    Inc(V);
  end;

  Inc(TotalCount, Count);
end;

// Compute whole-model AdamW statistics for the most recently completed update.
procedure GetAdamWStatistics(var WModelParams: TWModelParams; var WAdamWState: TWAdamWState;
  out ParamRMS, UpdateRMS, UpdateRatio, MRMS, SqrtVRMS: Double);
var
  k: Integer;
  TotalCount: Int64;
  SumParam2, SumUpdate2, SumM2, SumV: Double;
  Beta1Power, Beta2Power: Double;
begin
  ParamRMS := 0.0;
  UpdateRMS := 0.0;
  UpdateRatio := 0.0;
  MRMS := 0.0;
  SqrtVRMS := 0.0;

  if AdamWStep <= 0 then Exit;

  // Copy current parameters and AdamW moments to host.
  CopyParamsToHost(WModelParams);
  CopyAdamWStateToHost(WAdamWState);

  // AdamWStep is the number of updates already completed.
  Beta1Power := Power(AdamBeta1, AdamWStep);
  Beta2Power := Power(AdamBeta2, AdamWStep);

  SumParam2 := 0.0;
  SumUpdate2 := 0.0;
  SumM2 := 0.0;
  SumV := 0.0;
  TotalCount := 0;

  // Tied embeddings. Weight decay applies.
  AccumulateAdamWStatistics(@WModelParams.Embeddings.Value[0, 0], @WAdamWState.Embeddings.M[0, 0], @WAdamWState.Embeddings.V[0, 0],
    nVocab * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

  // Transformer blocks.
  for k := 0 to nBlock - 1 do begin

    // Attention weights. Weight decay applies.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wq.Value[0, 0], @WAdamWState.ParamBlock[k].Wq.M[0, 0], @WAdamWState.ParamBlock[k].Wq.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wk.Value[0, 0], @WAdamWState.ParamBlock[k].Wk.M[0, 0], @WAdamWState.ParamBlock[k].Wk.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Wv.Value[0, 0], @WAdamWState.ParamBlock[k].Wv.M[0, 0], @WAdamWState.ParamBlock[k].Wv.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W0.Value[0, 0], @WAdamWState.ParamBlock[k].W0.M[0, 0], @WAdamWState.ParamBlock[k].W0.V[0, 0],
      ModelDim * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // MLP weights. Weight decay applies.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W1.Value[0, 0], @WAdamWState.ParamBlock[k].W1.M[0, 0], @WAdamWState.ParamBlock[k].W1.V[0, 0],
      ModelDim * ModelDimProj, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].W2.Value[0, 0], @WAdamWState.ParamBlock[k].W2.M[0, 0], @WAdamWState.ParamBlock[k].W2.V[0, 0],
      ModelDimProj * ModelDim, WeightDecay, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // Biases. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].b1.Value[0], @WAdamWState.ParamBlock[k].b1.M[0], @WAdamWState.ParamBlock[k].b1.V[0],
      ModelDimProj, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].b2.Value[0], @WAdamWState.ParamBlock[k].b2.M[0], @WAdamWState.ParamBlock[k].b2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // LayerNorm 1. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Gamma1.Value[0], @WAdamWState.ParamBlock[k].Gamma1.M[0], @WAdamWState.ParamBlock[k].Gamma1.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Beta1.Value[0], @WAdamWState.ParamBlock[k].Beta1.M[0], @WAdamWState.ParamBlock[k].Beta1.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    // LayerNorm 2. No weight decay.
    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Gamma2.Value[0], @WAdamWState.ParamBlock[k].Gamma2.M[0], @WAdamWState.ParamBlock[k].Gamma2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);

    AccumulateAdamWStatistics(@WModelParams.ParamBlock[k].Beta2.Value[0], @WAdamWState.ParamBlock[k].Beta2.M[0], @WAdamWState.ParamBlock[k].Beta2.V[0],
      ModelDim, 0.0, Beta1Power, Beta2Power, SumParam2, SumUpdate2, SumM2, SumV, TotalCount);
  end;

  if TotalCount > 0 then begin
    ParamRMS := Sqrt(SumParam2 / TotalCount);
    UpdateRMS := Sqrt(SumUpdate2 / TotalCount);
    MRMS := Sqrt(SumM2 / TotalCount);

    // RMS(sqrt(V)) = sqrt(mean(V)).
    SqrtVRMS := Sqrt(SumV / TotalCount);

    if ParamRMS <> 0.0 then
      UpdateRatio := UpdateRMS / ParamRMS;
  end;
end;

procedure UpdateEmbeddingGradient(var WModelParams: TWModelParams; var WModelState: TWModelState);
begin
  with WModelParams do with WModelState do begin

    // Backpropagate through X = Embedding * Scale.
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dGrad);

    // Add input-side embedding gradient to the existing output-side
    // tied-embedding gradient.
    LaunchAddInputEmbeddingGrad(StateBlock[0].X.dGrad, Embeddings.dGrad, dInputTokens, SeqLen, ModelDim, nVocab);

    // Clip the complete tied-embedding gradient.
    LaunchClipVector(Embeddings.dGrad, nVocab * ModelDim, ClipLimit);
  end;
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

// Softmax Forward.
procedure LaunchSoftmaxForward(dIn: PSingle; dOut: PSingle; Rows: Integer; N: Integer; Temperature: Single);
begin
  LaunchSoftmaxForwardStrided(dIn, dOut, Rows, N, N, Temperature);
end;

end.

