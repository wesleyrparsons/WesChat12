unit Train;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  IOHandler,
  Math,
  Matrix,
  Outputhead,
  SysUtils,
  TransformForward,
  TransformBackprop,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState; const TokenizedCorpus: TIVector);

implementation

type
  TRowLossVector = array[0..SeqLen - 1] of Single;    // For computing CELoss on cuda.
var
  // Best model to save.
  AutoSaveEpoch: Integer;         // At this epoch, start saving new minimum loss models.
  MinSaveGap: Integer;
  MinSaveDelta: Double;
  // CE Loss.
  RowLoss: TRowLossVector;
  Beta1Power, Beta2Power: Single;

// Training log procs.
procedure OpenTrainLog(const FileName: string);
begin
  TrainLogFileName := FileName;

  AssignFile(TrainLogFile, TrainLogFileName);

  if FileExists(TrainLogFileName) then
    Append(TrainLogFile)
  else
    Rewrite(TrainLogFile);

  TrainLogOpen := True;

  Writeln(TrainLogFile);
  Writeln(TrainLogFile, '============================================================');
  Writeln(TrainLogFile, 'Training session started: ', DateTimeToStr(Now));
  Writeln(TrainLogFile, '============================================================');
  Flush(TrainLogFile);
end;

procedure TrainLog(const S: string);
begin
  if not TrainLogOpen then
    Exit;

  Writeln(TrainLogFile, DateTimeToStr(Now), '  ', S);
  Flush(TrainLogFile);
end;

procedure CloseTrainLog;
begin
  if not TrainLogOpen then
    Exit;

  Writeln(TrainLogFile, 'Training session ended: ', DateTimeToStr(Now));
  Writeln(TrainLogFile);
  Flush(TrainLogFile);

  CloseFile(TrainLogFile);
  TrainLogOpen := False;
end;

procedure LogTrainingStart(const ModelFileName: string);
begin
  if not TrainLogOpen then Exit;
  Writeln(TrainLogFile, 'Model file: ', ModelFileName);
  Writeln(TrainLogFile, 'Corpus ID: ', CorpusID);
  Writeln(TrainLogFile, 'Corpus bytes: ', nCorpus);
  Writeln(TrainLogFile, 'Raw tokens: ', RawTokenCount);
  Writeln(TrainLogFile, 'Tokenizer kind: ', Ord(TokenizerKind));
  Writeln(TrainLogFile, 'ModelDim: ', ModelDim);
  Writeln(TrainLogFile, 'ModelDimProj: ', ModelDimProj);
  Writeln(TrainLogFile, 'nBlock: ', nBlock);
  Writeln(TrainLogFile, 'nHead: ', nHead);
  Writeln(TrainLogFile, 'SeqLen: ', SeqLen);
  Writeln(TrainLogFile, 'nVocab: ', nVocab);
  Writeln(TrainLogFile, 'DimVocab: ', DimVocab);
  Writeln(TrainLogFile, 'Completed epochs: ', CompletedEpochs);
  Writeln(TrainLogFile, 'Global step: ', GlobalStep);
  Writeln(TrainLogFile, 'AdamW step: ', AdamWStep);
  Writeln(TrainLogFile, 'Learning rate: ', LearningRate:0:8);
  Writeln(TrainLogFile, 'Adaptive LR: ', AdaptiveLR);
  Writeln(TrainLogFile, 'Weight decay: ', WeightDecay:0:8);
  Writeln(TrainLogFile, 'Clip limit: ', ClipLimit:0:6);
  Writeln(TrainLogFile, 'Stride: ', Stride);
  Writeln(TrainLogFile, 'Attention dropout: ', ADropOut:0:4);
  Writeln(TrainLogFile, 'Residual dropout: ', RDropOut:0:4);
  Writeln(TrainLogFile, 'MLP dropout: ', MLPDropOut:0:4);
  if MinLoss < MaxDouble then begin
    Writeln(TrainLogFile, 'Historical minimum loss: ', MinLoss:0:6);
    Writeln(TrainLogFile, 'Historical minimum epoch: ', MinLossEpoch);
  end;
  if BestSavedLoss < MaxDouble then begin
    Writeln(TrainLogFile, 'Best saved loss: ', BestSavedLoss:0:6);
    Writeln(TrainLogFile, 'Best saved epoch: ', LastBestSaveEpoch);
  end;
  Writeln(TrainLogFile);
  Flush(TrainLogFile);
end;

procedure LogTrainingEpoch(const Epoch: Integer; const MEL, BPB, DiffLoss, LR: Double);
begin
  if not TrainLogOpen then Exit;
  Write(TrainLogFile, 'Epoch ', Epoch, ': MEL=', MEL: 0: 6, '; BPB=', BPB: 0: 6, '; DiffLoss=', DiffLoss: 0: 6, '; LR=', LR: 0: 8);
  if MinLoss < MaxDouble then
    Write(TrainLogFile, '; Best=', MinLoss: 0: 6, ' @ ', MinLossEpoch);
  Writeln(TrainLogFile);
  Flush(TrainLogFile);
end;

// Compute the CELoss with kernel routine.
function ComputeCELossGPU(dProbs: PSingle; dTargetTokens: PInteger; dRowLoss: PSingle; var RowLoss: TRowLossVector): Double;
var
  i: Integer;
begin
  LaunchCELossRows(dProbs, dTargetTokens, dRowLoss, SeqLen, nVocab, DimVocab);

  if DebugCudaChecks then
    CheckCudaError('Launch CELossRows.');

  cudaMemcpy(@RowLoss[0], dRowLoss, SeqLen * SizeOf(Single), cudaMemcpyDeviceToHost);

  if DebugCudaChecks then
    CheckCudaError('Copy CE row losses.');

  Result := 0.0;

  for i := 0 to SeqLen - 1 do begin
    if IsNan(RowLoss[i]) or IsInfinite(RowLoss[i]) then
      raise Exception.CreateFmt('ComputeCELossGPU: invalid row loss at position %d.', [i]);

    Result := Result + RowLoss[i];
  end;

  Result := Result / SeqLen;
end;

// Compute the CE loss only.
function ComputeLossOnly(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens, TargetTokens: TIDimVector): Double;
var
  Blk: Integer;
  RowLoss: TRowLossVector;
begin
  with WModelParams do with WModelState do begin

    // Make the procedure actually use its InputTokens parameter.
    cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);
    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic input tokens.');

    LaunchEmbeddingLookup(Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    for Blk := 0 to nBlock - 1 do begin
      RunTransformForward(WModelParams, WModelState, Blk);

      if Blk < nBlock - 1 then
        cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

    if DebugCudaChecks then
      CheckCudaError('Copy diagnostic target tokens.');

    RunOutputForward(WModelParams, WModelState);

    Result := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss, RowLoss);
  end;
end;

// Check for finite values.
procedure CheckFirstValuesFinite(const Name: string; Ptr: PSingle; Count: Integer);
var
  Temp: array[0..31] of Single;
  i, n: Integer;
begin
  n := Count;
  if n > 32 then n := 32;

  cudaMemcpy(@Temp[0], Ptr, n * SizeOf(Single), cudaMemcpyDeviceToHost);
  if DebugCudaChecks then
    CheckCudaError('copy ' + Name);

  for i := 0 to n - 1 do begin
    if (Temp[i] <> Temp[i]) or (Temp[i] > 1.0e20) or (Temp[i] < -1.0e20) then
      raise Exception.CreateFmt('BAD VALUE in %s[%d] = %g', [Name, i, Temp[i]]);
  end;
end;

// Create the target vector for use in head output.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Target[i] := TokenizedCorpus[StartIndex + i + 1];
end;

// Build the input vector .
procedure BuildInputVector(var Input: TIDimVector; const TokenizedCorpus: TIVector; const StartIndex, L: Integer);
var
  i: Integer;
begin
  for i := 0 to L - 1 do
    Input[i] := TokenizedCorpus[StartIndex + i];
end;

// Shuffle routines.
procedure BuildWindowStarts(const TokenCount, SeqLen, Stride, Epoch, StartStride: Integer; var Starts: TIVector);
var
  Start, Count: Integer;
begin
  SetLength(Starts, 0);

  Start := (Epoch * StartStride) mod Stride;
  Count := 0;

  while (Start + SeqLen + 1) <= TokenCount do begin
    SetLength(Starts, Count + 1);
    Starts[Count] := Start;
    Inc(Count);

    Inc(Start, Stride);
  end;
end;

// Compute sampled CE loss for use at initialing of training.
function ComputeSampledMeanLoss(var WModelParams: TWModelParams; var WModelState: TWModelState;
  const TokenizedCorpus: TIVector; const EpochForStarts, SampleCount: Integer): Double;
var
  i, Index, Start, Count: Integer;
  LossSum: Double;
  Starts: TIVector;
begin
  BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, EpochForStarts, StartStride, Starts);
  //BuildWindowStarts(RawTokenCount, SeqLen, Stride, EpochForStarts, StartStride, Starts);

  if Length(Starts) = 0 then begin
    Result := 0.0;
    Exit;
  end;

  LossSum := 0.0;
  Count := SampleCount;

  if Count > Length(Starts) then
    Count := Length(Starts);

  if Count = 1 then begin
    Start := Starts[0];

    BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);

    Result := ComputeLossOnly(WModelParams, WModelState, InputTokens, TargetTokens);
    Exit;
  end;

  for i := 0 to Count - 1 do begin
    Index := Round(Double(i) * High(Starts) / (Count - 1));
    Start := Starts[Index];

    BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);

    LossSum := LossSum +
      ComputeLossOnly(WModelParams, WModelState, InputTokens, TargetTokens);
  end;

  Result := LossSum / Count;
end;

// Helper routine for shuffle.
procedure Swap(var A, B: Integer);
var
  C: Integer;
begin
  C := A;
  A := B;
  B := C;
end;

// Shuffle starting points in tokenized corpus.
procedure ShuffleStarts(var Starts: TIVector);
var
  i, j: Integer;
begin
  if Length(Starts) <= 1 then Exit;

  for i := High(Starts) downto 1 do begin
    j := Random(i + 1);   // 0..i
    Swap(Starts[i], Starts[j]);
  end;
end;

// Initialize training settings for a new model.
procedure InitializeNewTrainingSettings;
begin
  GlobalStep := 0;
  AdamWStep := 0;
  CompletedEpochs := 1;
  GlobalSeed := 123456789;

  LearningStyle := SlowLearning;
  LearningRate := 0.000100;
  OverrideLearningRate := -1.0;
end;

// Run training.
procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; var WAdamWState: TWAdamWState; const TokenizedCorpus: TIVector);
const
  ln2 = 0.693147180559945309;
  MinWindowsPerEpoch = 100;
  BaselineSampleCount = 256;
var
  i, j, k: Integer;
  Blk, Epoch, FirstEpoch, OldLastBestSaveEpoch, Start, WindowCount, MaxStride: Integer;
  Loss, DiffLoss, BitsPerByte, EpochLoss, MeanEpochLoss, LastMeanEpochLoss, OldBestSavedLoss,
    TokensPerSecond, ElapsedEpochTime: Double;
  Starts: TIVector;
  NeedParamCopy, WasNewModel: Boolean;
  AdamParamRMS, AdamUpdateRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS: Double;
  EpochTime: TDateTime;
  EpochInterval: Integer = 10;
  CompactStats: TCompactTensorStats;
  AdaptiveLRReason: string = 'No change in adaptive learning yet.';

  // Report model specifications.
  procedure ReportModelSpecsInfo;
  begin
  // Writeln;
  Write('>>Model ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), ': nTC = ', Length(TokenizedCorpus), '; nCorpus =  ', nCorpus, '; nVocab = ', nVocab,
    '; DimVocab = ', DimVocab, '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock =  ', nBlock,
    '; Proj = ', Proj, '; Shuffle = ', ShuffleWindows, '; Tok Kind = ', TokenizerKindName(TokenizerKind), '; DropOut = ', Training);
  if Training then
    Writeln(' (', ADropOut: 4: 3, ' ', MLPDropOut: 4: 3, ' ', RDropOut: 4: 3, ').')
  else
    Writeln('.');
  end;

  // Report AdamW statistics.
  procedure AdamWReportInfo;
  begin
    Writeln('>>AdamW: Step = ', AdamWStep, '; Param RMS = ', AdamParamRMS: 0: 8, '; Update RMS = ', AdamUpdateRMS: 0: 10,
    '; Update = ', 100.0 * AdamUpdateRatio: 0: 5, '%; M RMS = ', AdamMRMS: 0: 8, '; Sqrt(V) RMS = ', AdamSqrtVRMS: 0: 8, '.');
  end;

  // Report options in training.
  procedure ReportOptions;
  begin
    Writeln('>>', DateTimeToStr(Now), '  D = set Dropout rate. I = get program Info. L = set Learning rate. N = go to iNference. P = Pause. S = Save model. ',
      'T = set sTride. V = toggle Verbose mode. W = set Weight decay. X = eXit training.');
  end;

  // Report epoch information.
  procedure ReportEpochInfo;
  begin
    Write(TimeToStr(Now), ' Epoch ', Epoch, ' ended. Steps = ', GlobalStep - WindowCount + 1, '..', GlobalStep, '. Best epoch = ', MinLossEpoch, '. LR = ', LearningRate: 9: 7,
      '. Mean loss = ',  MeanEpochLoss: 8: 6, '; Perplexity = ', exp(MeanEpochLoss): 8: 6, '; Bits per byte = ', BitsPerByte: 0: 4);
    Write('; Loss ');
    if DiffLoss = 0 then
      Writeln('same.')
    else if DiffLoss > 0 then
      Writeln('better by ', DiffLoss: 8: 6, '.')
    else
      Writeln('worse by ', -DiffLoss: 8: 6, '.');
  end;

  // Display training and LR information.
  procedure EpochIntervalReportInfo;
  begin
    Write('Window # = ', WindowCount);
    if Epoch > 0 then
      Writeln('; Epoch time = ', Trunc(ElapsedEpochTime) div 3600,
        ' hours, ', (Trunc(ElapsedEpochTime) mod 3600) div 60, ' minutes, ', (ElapsedEpochTime mod 60): 0: 4,
        ' seconds; Training speed = ', TokensPerSecond: 0: 0, ' tokens/second.')
    else
      Writeln('.');
  end;

  function TrainReadIfKeyPressed: Boolean;
  var
    key: char;
    j: Integer;
    Dropout: Single;
  begin
    Result := False;
    key := CheckForControlKey;
    case key of
      'd', 'D': begin        // Change dropout rates.
        Write('Enter dropout rates: ');
        Readln(DropOut);
        ADropout := DropOut;
        MLPDropout := DropOut;
        RDropout := DropOut;
        Writeln('New dropout rates = ', Dropout: 4: 3, '. Training...');
      end;
      'i', 'I': begin        // Display training info.
        ReportProgramInfo;
        // Write('>>');
        EpochIntervalReportInfo;
        ReportOptions;
        PauseNNL;
      end;
      'l', 'L': begin        // Override learning rate.
        Write('Enter override learning rate: ');
        Readln(OverrideLearningRate);
        Writeln('New learning rate = ', OverrideLearningRate: 9: 7, '. Training...');
      end;
      'n', 'N': begin        // Exit training. Go to inference.
        Writeln('Stopping training. Going to inference...');
        TrainSuccess := True;
        Result := True;
        StopTraining := True;
      end;
      'p', 'P': begin        // Pause work.
        Write('Paused... ');
        PauseNNL;
      end;
      's', 'S': begin        // Save model.
        Write('Enter model filename, blank for automatic checkpoint: ');
        Readln(ModelFileName);

        ModelFileName := Trim(ModelFileName);

        if ModelFileName = '' then begin
          if Trim(WorkingName) = '' then WorkingName := 'weschat';

          ModelFileName := ModelDir + WorkingName + '_epoch' + IntToStr(Epoch) +
            '_step' + IntToStr(GlobalStep) + '_' + FormatDateTime('yyyy-mm-dd_hhnss', Now) + '.model';
        end
        else begin
          if ExtractFileExt(ModelFileName) = '' then
            ModelFileName := ModelFileName + '.model';

          if not IsAbsolutePath(ModelFileName) then
            ModelFileName := ModelDir + ModelFileName
          else
            ModelFileName := ExpandFileName(ModelFileName);
        end;

        if SaveModel(ModelFileName, WModelParams, WAdamWState) then begin
          ModelPresent := True;
          Writeln('Model saved. File = ', ModelFileName, '.');
        end
        else
          Writeln('File not saved. Training...');
      end;
      't', 'T': begin
        Write('Current stride = ', Stride, '. Enter new stride: ');
        Readln(j);
        MaxStride := (Length(TokenizedCorpus) - SeqLen) div MinWindowsPerEpoch;

        if j <= 0 then
          Writeln('Stride must be greater than zero.')
        else if j > MaxStride then
          Writeln('Stride is too large. Maximum stride for at least ',
            MinWindowsPerEpoch, ' windows per epoch is ', MaxStride, '.')
        else begin
          Stride := j;
          Writeln('New stride = ', Stride, '. It will take effect next epoch.');
        end;
      end;
      'v', 'V': begin        // Enable verbose transform.
        VerboseTransform := not VerboseTransform;
        Writeln('Verbose transform mode: ', VerboseTransform, '. Training...');
      end;
      'w', 'W': begin        // Override weight decay.
        Write('Enter new weight decay: ');
        Readln(WeightDecay);
        Writeln('New weight decay = ', WeightDecay: 8: 6, '. Training...');
      end;
      'x', 'X': begin        // Exit training. Success, go to main menu.
        Writeln('Exit requested. Stopping training.');
        TrainSuccess := False;
        Result := True;
        StopTraining := True;
      end;
    end;
  end;

begin     writeln('start training ', TokenizerKindName(TokenizerKind)); pause;
  // For saving models.
  LastBestSaveEpoch := -1000000000;    // Sentinel for no best model saved.
  AutoSaveEpoch := 10;                 // Start auto save of bext model at this epoch.
  BestSavedLoss := MaxDouble;          // The prior best saved loss.
  MinSaveGap := 5;                     // Wait this many epochs to save a best model again.
  MinSaveDelta := 0.00025;             // Smallest loss improvement needed for saving best model.

  // Getting working name right.
  if Trim(WorkingName) = '' then begin
    Writeln('WorkingName is blank. Using "weschat".');
    WorkingName := 'weschat';
  end;

  // Start logging.
  OpenTrainLog(LogDir + WorkingName + '_train.log');
  LogTrainingStart(BestModelFileName);

  // Initializing each epoch's loss.
  MeanEpochLoss := 0;
  EpochTime := Now;
  ElapsedEpochTime := 0.0;
  MinLoss := 1000000;
  MinLossEpoch := -1;

  // Initializing at zero the speed and efficiency statistics.
  TokensPerSecond := 0.0;

  // General initialization.
  StopTraining := False;
  Training := True;
  WasNewModel := NewModel;
  InitializeAdaptiveLRState(AdaptiveLRState);
  if WasNewModel then
    InitializeNewTrainingSettings;

  // Guard for empty nCorpus.
  if nCorpus = 0 then begin
    nCorpus := 1000000;
    Writeln('Empty corpus. nCorpus set at 1000000.');
    Pause;
  end;

  // Check DimVocab is large enough.
  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting training...');
    TrainSuccess := False;
    Exit;
  end;

  // Initializing state.
  InitializeTransformerState(WModelState);

  // Initializing params if new model.
  NeedParamCopy := (not CudaAllocated) or ParamsNeedCopyToDevice or WasNewModel;
  if WasNewModel then
    InitializeTransformerParams(WModelParams);
  NewModel := False;

  // Initiate CUDA and allocate all CUDA buffers, including AdamW dM and dV.
  StartCuda(WModelParams, WModelState, WAdamWState);
  Writeln('--- Training ---');

  try
    if NeedParamCopy then begin
      CopyParamsToDevice(WModelParams);
      ParamsNeedCopyToDevice := False;
    end;

    // Send InvFreq to device.
    CopyInvFreqToDevice(WModelState);

    // Now dM and dV exist, so initialize or restore AdamW.
    if WasNewModel then begin
      InitializeWAdamWState(WAdamWState);
      AdamWStep := 0;
      AdamWStateLoaded := True;
    end
    else if AdamWStateLoaded then begin
      // WES2 model containing saved AdamW state.
      CopyAdamWStateToDevice(WAdamWState);
    end
    else begin
      // WES1 model: loaded weights, but no AdamW history.
      InitializeWAdamWState(WAdamWState);
      AdamWStep := 0;
      AdamWStateLoaded := True;
    end;

    // Initialize epoch/sequence loop.
    Start := 0;
    FirstEpoch := CompletedEpochs;

    // Estimate baseline loss from evenly spaced windows across the corpus.
    LastMeanEpochLoss := ComputeSampledMeanLoss(WModelParams, WModelState, TokenizedCorpus, FirstEpoch, BaselineSampleCount);
    // Reports before training.
    Writeln('>>Estimated baseline loss before training = ', LastMeanEpochLoss: 8: 6, '; Perplexity = ', Exp(LastMeanEpochLoss): 0: 7,
      '; Bits per byte = ', LastMeanEpochLoss * RawTokenCount / nCorpus / Ln2: 0: 7, '.');
    ReportModelSpecsInfo;
    ReportOptions;
    Writeln('Training started...');

    // Display embeddings.
    if VerboseTransform then begin
      Writeln('Display Embeddings.Value prior to Transform.');
      DisplayX(WModelParams.Embeddings.Value, B);
      Pause;
    end;

    // Loop through epochs.
    for Epoch := FirstEpoch to MaxEpochs - 1 do with WModelState do begin
      EpochLoss := 0;
      WindowCount := 0;

      // Build a window start point.
      BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, Epoch, StartStride, Starts);

      // Shuffle the window.
      if ShuffleWindows then
        ShuffleStarts(Starts);

      // Stride loop thru Sequence.
      for i := 0 to High(Starts) do begin
        Start := Starts[i];

        if TrainReadIfKeyPressed then begin
          StopTraining := True;
          Break;
        end;

        // Display number of loops thru embed loop.
        if DisplayWindow then
          Writeln('Epoch is ', Epoch, ' SeqLen window is from ', Start, ' to ', Start + SeqLen, ' on loop ', WindowCount);

        // Build the target vector, one ahead, for the loss stage.
        BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);
        cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

        // Build the input vector.
        BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
        cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

        // Checking tokens.
        if VerboseTransform then begin
          Writeln('Checking tokens');
          for j := 0 to SeqLen - 1 do
            Write(j: 4, ' token=', InputTokens[j], ' ');
          Writeln('Values for row zero.');
          k := InputTokens[0];
          for j := 0 to 15 do
            Write(WModelParams.Embeddings.Value[k, j]: 8: 5, ' ');
         Pause;
        end;

        // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
        LaunchEmbeddingLookup(WModelParams.Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);

        // Display X.Value matrix.
        if VerboseTransform then begin
          cudaMemcpy(@WModelState.StateBlock[0].X.Value[0,0], StateBlock[0].X.dValue, XSize, cudaMemcpyDeviceToHost);
          VTPDisplayX('Display X.Value before transform.', StateBlock[0].X.Value, G);
          VTPDisplayX('Display X.Value before transform.', StateBlock[0].X.Value, B);
        end;

        // Optional transformer-style embedding scaling by sqrt(d_model).
        CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

        // Zero gradients.
        for k := 0 to nBlock - 1 do
          ZeroGradients(WModelParams, WModelState, k);
        if DebugCudaChecks then
          CheckCudaError('Zero transformer state and parameter gradients.');

        // Forward pass thru transformer.
        for Blk := 0 to nBlock - 1 do begin
          RunTransformForward(WModelParams, WModelState, Blk);

          if Blk < nBlock - 1 then
            // CopyXTensor(StateBlock[Blk].X7, StateBlock[Blk + 1].X);
            cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
        end;

        // Run the output-head forward pass.
        RunOutputForward(WModelParams, WModelState);

        // Compute mean cross-entropy loss on the GPU and copy only SeqLen losses.
        Loss := ComputeCELossGPU(dProbs, dTargetTokens, dRowLoss, RowLoss);

        // Compute the logit gradient, X7 gradient, and output-side tied-embedding gradient.
        RunOutputBackward(WModelParams, WModelState);

        if VerboseTransform then
          Writeln('            Switch from Forward to Backprop');

        // Backprop pass thru transformer.
        for Blk := nBlock - 1 downto 0 do with WModelState do begin
          RunTransformBackprop(WModelParams, WModelState, Blk);

          if Blk > 0 then
            cudaMemcpy(StateBlock[Blk - 1].X7.dGrad, StateBlock[Blk].X.dGrad, XSize, cudaMemcpyDeviceToDevice);
        end;

        // Clip vectors.
        for k := 0 to nBlock - 1 do
          with WModelParams.ParamBlock[k] do begin
            LaunchClipVector(Wq.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(Wk.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(Wv.dGrad, ModelDim * ModelDim, ClipLimit);
            LaunchClipVector(W0.dGrad, ModelDim * ModelDim, ClipLimit);

            LaunchClipVector(W1.dGrad, ModelDim * ModelDimProj, ClipLimit);
            LaunchClipVector(W2.dGrad, ModelDimProj * ModelDim, ClipLimit);

            LaunchClipVector(b1.dGrad, ModelDimProj, ClipLimit);
            LaunchClipVector(b2.dGrad, ModelDim, ClipLimit);

            LaunchClipVector(Gamma1.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Beta1.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Gamma2.dGrad, ModelDim, ClipLimit);
            LaunchClipVector(Beta2.dGrad, ModelDim, ClipLimit);
          end;
        if DebugCudaChecks then
          CheckCudaError('Gradient clipping.');

        if VerboseTransform then begin
          cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);

          if DebugCudaChecks then
            CheckCudaError('Copy probabilities for verbose display.');

          VTPDisplayX('Display Probs after softmax.', Probs, B);
        end;

        // Set learning rate.
        // Manual override has highest priority.
        if OverrideLearningRate <> -1.0 then
          LearningRate := OverrideLearningRate

        // Adaptive learning owns LearningRate once enabled.
        else if not AdaptiveLR then begin
          // Use schedule to calculate learning rate.
          Case LearningStyle of
            // Flat AdamW schedule.
            FlatLearning: LearningRate := 0.0003;

            // Slow AdamW schedule.
            SlowLearning: case Epoch of
              1..2:       LearningRate := 0.000100;
              3..7:       LearningRate := 0.000075;
              8..15:      LearningRate := 0.000050;
              16..30:     LearningRate := 0.000025;
              31..60:     LearningRate := 0.000015;
              61..100:    LearningRate := 0.000010;
              else        LearningRate := 0.000005;
            end;

            // Fast AdamW schedule.
            FastLearning: case Epoch of
              1..2:       LearningRate := 0.00030;
              3..10:      LearningRate := 0.00020;
              11..30:     LearningRate := 0.00010;
              31..100:    LearningRate := 0.000050;
              101..400:   LearningRate := 0.000025;
              401..800:   LearningRate := 0.000010;
              else        LearningRate := 0.000005;
            end;
          end;
        end;

        // Do AdamW optimization.
        Beta1Power := Single(Power(AdamBeta1, AdamWStep + 1));
        Beta2Power := Single(Power(AdamBeta2, AdamWStep + 1));

        // Update embedding gradients with AdamW.
        UpdateEmbeddingGradient(WModelParams, WModelState);

        // Update transformer block parameters with AdamW.
        for k := 0 to nBlock - 1 do
          AdamWOptimizeBlock(WModelParams, WAdamWState, k, Beta1Power, Beta2Power);

        // Update tied embeddings.
        AdamWOptimizeEmbeddings(WModelParams, WAdamWState, Beta1Power, Beta2Power);
        Inc(AdamWStep);

        // Diagnostic check for excessive embeddings values.
        if DebugCudaChecks then begin
          CheckFirstValuesFinite('Embeddings', WModelParams.Embeddings.dValue, nVocab * ModelDim);
          for k := 0 to nBlock - 1 do with WModelParams.ParamBlock[k] do begin
            CheckFirstValuesFinite('Wq', Wq.dValue, ModelDim * ModelDim);
            CheckFirstValuesFinite('W1', W1.dValue, ModelDim * ModelDimProj);
            CheckFirstValuesFinite('W2', W2.dValue, ModelDimProj * ModelDim);
            CheckFirstValuesFinite('Gamma1', Gamma1.dValue, ModelDim);
            CheckFirstValuesFinite('Gamma2', Gamma2.dValue, ModelDim);
          end;
          CheckCudaError('Training parameter update.');
        end;

        // Update sequence loop.
        EpochLoss := EpochLoss + Loss;
        Inc(WindowCount);
        Inc(GlobalStep);
      end;    // End sequence loop.

      if StopTraining then Exit;

      // Compute mean epoch loss.
      if WindowCount = 0 then begin
        Writeln('Epoch ', Epoch, ' contained no training windows.');
        Continue;
      end;
      MeanEpochLoss := EpochLoss / WindowCount;
      DiffLoss := LastMeanEpochLoss - MeanEpochLoss;
      LastMeanEpochLoss := MeanEpochLoss;

      CompletedEpochs := Epoch + 1;
      BitsPerByte := MeanEpochLoss * RawTokenCount / nCorpus / Ln2;

      // Save best model subject to conditions.
      if MeanEpochLoss < MinLoss then begin
        MinLoss := MeanEpochLoss;
        MinLossEpoch := Epoch;

        if (Epoch >= AutoSaveEpoch) and
           ((Epoch - LastBestSaveEpoch) >= MinSaveGap) and
           ((BestSavedLoss = MaxDouble) or ((BestSavedLoss - MeanEpochLoss) >= MinSaveDelta)) then begin

          // Display saving of first and subsequent best models.
          if LastBestSaveEpoch < 0 then
            Write('--Saving first best model in epoch ', Epoch)
          else
            Write('--Saving new best model. Previous saved best = ',
              BestSavedLoss:9:7, ' in epoch ', LastBestSaveEpoch,
              '; new best = ', MeanEpochLoss:9:7, ' in epoch ', Epoch);

          OldLastBestSaveEpoch := LastBestSaveEpoch;
          OldBestSavedLoss := BestSavedLoss;

          // Set before SaveModel so these values are included in the checkpoint.
          LastBestSaveEpoch := Epoch;
          BestSavedLoss := MeanEpochLoss;

          if SaveModel(BestModelFileName, WModelParams, WAdamWState) then begin
            Writeln('. Best model saved: ', BestModelFileName, '.');

            TrainLog('Best model saved. Epoch=' + IntToStr(Epoch) +
              '; MEL=' + FloatToStrF(MeanEpochLoss, ffFixed, 10, 6) +
              '; File=' + BestModelFileName);
          end
          else begin
            // Save failed, so restore the previous saved-best state.
            LastBestSaveEpoch := OldLastBestSaveEpoch;
            BestSavedLoss := OldBestSavedLoss;

            Writeln('. Best model not saved.');
            TrainLog('Best model save FAILED. Epoch=' + IntToStr(Epoch) +
              '; MEL=' + FloatToStrF(MeanEpochLoss, ffFixed, 10, 6) +
              '; File=' + BestModelFileName);
          end;
        end;
      end;

      // Compute improvement.
      if DiffLoss >= 0 then       // Loss gets better or stays same.
        Write('^^')
      else if DiffLoss < 0 then
        Write('vv');              // Loss gets worse.

      // Report the line info for the epoch.
      ReportEpochInfo;

      // Display loss progress every EpochInterval epochs.
      if (Epoch mod EpochInterval) = 0 then begin

        // Timing epochs.
        ElapsedEpochTime := (Now - EpochTime) * 86400.0 / EpochInterval;
        EpochTime := Now;

        ReportModelSpecsInfo;

        // Compute and report AdamW.
        GetAdamWStatistics(WModelParams, WAdamWState, AdamParamRMS, AdamUpdateRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS);
        AdamWReportInfo;

        // Display learning-rate mode and setting.
        if OverrideLearningRate <> -1.0 then begin
          Writeln('>>Learning rate: Override = ', LearningRate: 8: 6, '.');
        end
        else if AdaptiveLR then begin
          Writeln('>>Learning rate: Adaptive = ', LearningRate: 9: 7, '. ', AdaptiveLRReason);
        end
        else begin
          Case LearningStyle of
            FlatLearning: Write('>>Learning rate: Flat = ', LearningRate: 9: 7, '.');

            SlowLearning: Write('>>Learning rate: Slow = ', LearningRate: 9: 7, ' with 1..2: 0.000100; 3..7: 0.000075; ',
              '8..15: 0.000050; 16..30: 0.000025; 31..60: 0.000015; 61..100: 0.000010; else 0.000005.');

            FastLearning: Write('>>Learning rate: Fast = ', LearningRate: 9: 7, ' with 1..2: 0.00030; 3..10: 0.00020; ',
              '11..30: 0.00010; 31..100: 0.000050; 101..400: 0.000025; 401..800: 0.000010; else 0.000005.');

            RolledOffLearning: Write('>>Learning rate: Rolloff = ', LearningRate: 9: 7,
              ' Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
          end;
          Writeln(' WeightDecay = ', WeightDecay: 6: 4, '; Clip = ', ClipLimit: 6: 4, '.');
        end;

        // Speed and efficiency statistics.
        if ElapsedEpochTime > 0.0 then begin
          TokensPerSecond := (WindowCount * SeqLen) / ElapsedEpochTime;
        end;

        // Display training and LR information, and reset running loss.
        EpochIntervalReportInfo;

        // Report full tensor stats.
        if VerboseTransform then
          ReportAdamWTensorStatistics(WModelParams, WAdamWState);
        // Report compact tensor stats.
        ReportCompactTensorStatistics(WModelParams, CompactStats);

        // Compute and display new adaptive LR.
        with CompactStats do
          if AdaptiveLR and (OverrideLearningRate = -1.0) then
            ApplyAdaptiveLR(AdaptiveLRState, LearningRate, FloorLearningRate, MeanEpochLoss, MinLoss, DiffLoss,
            AdamParamRMS, AdamUpdateRatio, AdamMRMS, AdamSqrtVRMS, MaxGammaRMS, Epoch, AdaptiveLRReason);
      end;      // End epoch loss display.

      // Log training data.
      LogTrainingEpoch(CompletedEpochs, MeanEpochLoss, BitsPerByte, DiffLoss, LearningRate);
    end    // End epoch.
  except
    on E: Exception do begin
      TrainSuccess := False;
      Writeln('Training Error: ', E.ClassName, '; ', E.Message, '.');
      Pause;
      Exit;
    end;
  end;

  Writeln;
  CloseTrainLog;
  Writeln('Training ended.');
  TrainSuccess := True;
  Pause;
end;

end.

