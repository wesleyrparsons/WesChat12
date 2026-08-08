unit Train;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

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

procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; const TokenizedCorpus: TIVector);

implementation

const
  Scale = Sqrt(ModelDim);         // Transformer-style embedding scaling by sqrt(d_model).
  MinWindowsPerEpoch = 100;       // Need for stride, especially changing.
type
  TRowLossVector = array[0..SeqLen - 1] of Single;    // For computing CELoss on cuda.
var
  LastBestSaveEpoch: Integer;     // Saving models.
  AutoSaveEpoch: Integer;         // At this epoch, start saving new minimum loss models.
  BestSavedLoss: Double;
  MinSaveGap: Integer;
  MinSaveDelta: Double;
  RowLoss: TRowLossVector;

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

    {RunOutputForward(WModelParams, WModelState);
    cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
    if DebugCudaChecks then
      CheckCudaError('Copy probabilities for diagnostic loss.');

    Result := ComputeCELoss(Probs, TargetTokens);}
  end;
end;

// Check for finite values.
procedure CheckFirstValuesFinite(const Name: string; Ptr: PSingle; Count: Integer);
var
  Temp: array[0..31] of Single;
  i, N: Integer;
begin
  N := Count;
  if N > 32 then N := 32;

  cudaMemcpy(@Temp[0], Ptr, N * SizeOf(Single), cudaMemcpyDeviceToHost);
  if DebugCudaChecks then
    CheckCudaError('copy ' + Name);

  for i := 0 to N - 1 do begin
    if (Temp[i] <> Temp[i]) or
       (Temp[i] > 1.0e20) or
       (Temp[i] < -1.0e20) then begin
      Writeln('BAD VALUE in ', Name, '[', i, '] = ', Temp[i]:12:6);
      Pause;
      Halt;
    end;
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

procedure Swap(var A, B: Integer);
var
  C: Integer;
begin
  C := A;
  A := B;
  B := C;
end;

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

// Run the training.
procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; const TokenizedCorpus: TIVector);
var
  i, j, k: Integer;
  Blk, Epoch, FirstEpoch, Start, WindowCount, MinLossEpoch: Integer;
  MaxStride: Integer;
  Loss, MinLoss, DiffLoss, LastLoss, EpochLoss, MEL, StartLoss: Double;
  RecentImprovementIndex, RecentImprovementCount: Integer;
  RecentImprovementSum, MeanRunningImprovement: Double;
  RecentImprovements: array[0..RecentCount - 1] of Double;
  Starts: TIVector;
  NeedParamCopy, WasNewModel: Boolean;

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
        Writeln('Training. Work = ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), '; nTC = ', Length(TokenizedCorpus), ',  nVocab = ', nVocab,
          '; DimVocab = ', DimVocab, '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock,
          '; Proj = ', Proj, '; DropOut = ', Training, '; Shuffling = ', ShuffleWindows, '.');
        Write(DateTimeToStr(Now), '  D = set Dropout rate. I = get program Info. L = set Learning rate. N = go to iNference. P = Pause. S = Save model.');
        Writeln('T = set sTride. V = toggle Verbose mode. W = set Weight decay. X = eXit training. Training...');
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
        Writeln('Paused...');
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

        if SaveModel(ModelFileName, WModelParams) then begin
          ModelPresent := True;
          Write('Saving model File = ', ModelFileName);
          Write('; Epoch = ', Epoch);
          Write('; GlobalStep = ', GlobalStep);
          Write('; LearningRate = ', LearningRate: 0: 7);
          Write('; WeightDecay = ', WeightDecay: 0: 7);
          Write('; Current loss = ', MEL: 0: 7);
          Writeln('; Perplexity = ', Exp(MEL): 0: 7, '.');
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

begin
  // For saving models.
  LastBestSaveEpoch := -1000000000;
  AutoSaveEpoch := 50;
  BestSavedLoss := MaxDouble;
  MinSaveGap := 10;
  MinSaveDelta := 0.01;      // Adjust for saving best model.

  // Getting working name right.
  if Trim(WorkingName) = '' then begin
    Writeln('WorkingName is blank. Using "weschat".');
    WorkingName := 'weschat';
  end;
  Writeln('Automatic best model file: ', ModelDir + WorkingName + '_best.model');

  // For each epoch's loss.
  MEL := 0;
  MinLoss := 1000000;
  MinLossEpoch := -1;

  // Initialization.
  StopTraining := False;
  Training := True;
  WasNewModel := NewModel;

  if WasNewModel then begin
    GlobalStep := 0;
    CompletedEpochs := 0;
    GlobalSeed := 123456789;
  end;

  // Initialize rolling improvement variables.
  RecentImprovementIndex := 0;
  RecentImprovementCount := 0;
  RecentImprovementSum := 0.0;
  MeanRunningImprovement := 0.0;
  LastLoss := 0.0;
  for i := 0 to High(RecentImprovements) do
    RecentImprovements[i] := 0.0;

  // Check DimVocab is large enough.
  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting training...');
    TrainSuccess := False;
    Exit;
  end;

  // Initialize state.
  InitializeTransformerState(WModelState);

  // Initialize params if new model.
  NeedParamCopy := (not CudaAllocated) or ParamsNeedCopyToDevice or NewModel;
  if WasNewModel then
    InitializeTransformerParams(WModelParams);
  NewModel := False;

  // Initiate Cuda.
  StartCuda(WModelParams, WModelState);

  try
    if NeedParamCopy then begin
      CopyParamsToDevice(WModelParams);
      ParamsNeedCopyToDevice := False;
    end;

    // Send InvFreq to device.
    CopyInvFreqToDevice(WModelState);

    // Initialize epoch/sequence loop.
    Start := 0;
    FirstEpoch := CompletedEpochs;

    Writeln('Training started.');
    Writeln;
    Write(DateTimeToStr(Now), '  I = get program Information. L = set Learning rate. N = go to iNference. P = Pause. ');
    Writeln('S = Save. T = set sTride. V = toggle Verbose mode. X = eXit training. Training...');

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

      // Start := (Epoch * StartStride) mod Stride;
      // Build a window start point.
      BuildWindowStarts(Length(TokenizedCorpus), SeqLen, Stride, Epoch, StartStride, Starts);

      // Shuffle the window.
      if ShuffleWindows then
        ShuffleStarts(Starts);

      // Stride loop thru Sequence.
      // while ((Start + SeqLen + 1) <= Length(TokenizedCorpus)) do begin
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
        // BuildInputMatrix(WModelState.StateBlock[0].X.Value, InputTokens, TokenizedCorpus, WModelParams, Start, SeqLen);
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

        // Use schedule to calculate learning rate. Remember Training may be False (affects dropouts).
        Case LearningStyle of
          // Flat schedule of 0.01;.
          FlatLearning: LearningRate := 0.01;
          // Slow schedule.
          SlowLearning:
            case Epoch of
              0..10:      LearningRate := 0.01;
              11..20:     LearningRate := 0.005;
              21..100:    LearningRate := 0.0005;
              101..1000:  LearningRate := 0.0001;
              else        LearningRate := 0.00005;
            end;
          // Fast schedule.
          FastLearning:
            case Epoch of
              0..30:      LearningRate := 0.01;
              31..100:    LearningRate := 0.005;
              101..400:   LearningRate := 0.001;
              401..800:   LearningRate := 0.0005;
              else        LearningRate := 0.0001;
            end;
          // Rolled off schedule.
          RolledOffLearning: begin
            // LearningRate := Max(FloorLearningRate, BaseLearningRate * Power(Rolloff, GlobalStep));
            LearningRate := FloorLearningRate + (BaseLearningRate - FloorLearningRate) * Power(Rolloff, GlobalStep);
          end;
        end;

        // User can set override learning rate.
        if OverrideLearningRate <> -1.0 then
          LearningRate := OverrideLearningRate;

        // Decay tied to learning rate, AdamW-style.
        DecayScale:= 1.0 - LearningRate * WeightDecay;

        // Modify weights and biases.
        for k := 0 to nBlock - 1 do
          Optimization(WModelParams, k);

        // Apply the total embedding gradient (output-side + input-side).
        UpdateEmbeddings(wModelParams, WModelState);

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

      // Compute minimum epoch loss.
      if WindowCount = 0 then begin
        Writeln('Epoch ', Epoch, ' contained no training windows.');
        Continue;
      end;

      MEL := EpochLoss / WindowCount;

      // Difference from previous epoch. Positive DiffLoss means improvement.
      if Epoch = FirstEpoch then begin
        DiffLoss := 0.0;
        StartLoss := MEL;
      end
      else
        DiffLoss := LastLoss - MEL;

      LastLoss := MEL;

      // Rolling mean improvement over the last RecentCount epoch transitions.
      // Positive means loss decreased; negative means loss increased.
      if Epoch > 0 then begin
        if RecentImprovementCount < RecentCount then
          Inc(RecentImprovementCount)
        else
          RecentImprovementSum := RecentImprovementSum - RecentImprovements[RecentImprovementIndex];

        RecentImprovements[RecentImprovementIndex] := DiffLoss;
        RecentImprovementSum := RecentImprovementSum + DiffLoss;
        RecentImprovementIndex := (RecentImprovementIndex + 1) mod RecentCount;

        MeanRunningImprovement := RecentImprovementSum / RecentImprovementCount;
      end;

      CompletedEpochs := Epoch + 1;

      // Compute StartLoss.
      if Epoch = 0 then
        StartLoss := MEL;

      // Save best model subject to conditions.
      if MEL < MinLoss then begin
        MinLoss := MEL;
        MinLossEpoch := Epoch;

        if (Epoch >= AutoSaveEpoch) and ((Epoch - LastBestSaveEpoch) >= MinSaveGap) and
          ((BestSavedLoss = MaxDouble) or ((BestSavedLoss - MEL) >= MinSaveDelta)) then begin

            // Display saving of first and subsequent best models.
            if LastBestSaveEpoch < 0 then   // First time.
              Write('Saving first best model in epoch ', Epoch)
            else                            // Subsequent times.
              Write('Saving new best model. Previous saved best = ', BestSavedLoss: 9: 7, ' in epoch ', LastBestSaveEpoch, '; new best = ', MEL: 9: 7, ' in epoch ', Epoch);
            // For saving all best models.
            if SaveModel(ModelDir + WorkingName + '_best.model', WModelParams) then begin
              LastBestSaveEpoch := Epoch;
              BestSavedLoss := MEL;
              Writeln('. Best model saved.');
            end
            else
              Writeln('. Best model not saved.');
        end;
      end;

      if MinLossEpoch = Epoch then
        Write('^^')
      else
        Write('--');
      Write('Epoch ', Epoch, ' ended. Steps = ', GlobalStep - WindowCount + 1, '..', GlobalStep, '.');
      Write(' LR = ', LearningRate:7 : 5, '. Mean loss: Start = ', StartLoss: 8: 6, '; Min = ',
        MinLoss: 8: 6, ' in epoch ', MinLossEpoch, '; Current = ', MEL: 8: 6, '; Perplexity = ', exp(MEL): 8: 6,
        '; Rolling improvement', RecentImprovementCount, ' = ', MeanRunningImprovement: 9: 7);
      if Epoch > 0 then begin
        if DiffLoss > 0 then
          Write('; Better by ', DiffLoss: 8: 6)
        else
          Write('; Worse by ', -DiffLoss: 8: 6);
      end;
      Writeln('.');

      // Display loss progress every 10 epochs.
      if (Epoch > 0) and ((Epoch mod 10) = 0) then begin

        // Parameters.
        Write('>> Work = ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), '; nTC = ', Length(TokenizedCorpus), '; nVocab = ', nVocab, '; DimVocab = ', DimVocab,
          '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock, '; Proj = ', Proj, '; Shuffling = ', ShuffleWindows,
          '; DropOut = ', Training);
        if Training then
          Writeln(' (', ADropOut: 4: 3, ' ', MLPDropOut: 4: 3, ' ', RDropOut: 4: 3, ').')
        else
          Writeln('.');

        if OverrideLearningRate = -1.0 then begin
          // Not override learning rate.
          Case LearningStyle of
            FlatLearning:
              // Display flat learning rate.
              Write('>> Learning rate (flat) = ', LearningRate: 9: 7, '.');
            SlowLearning:
              // Display slow learning rate schedule.
              Write('>> Learning rate (slow) = ', LearningRate: 9: 7, ' with 0..10: 0.01; 11..20: 0.005; 21..100: 0.0005; 101..1000: 0.0001; else 0.00005.');
            FastLearning:
              // Display fast learning rate schedule.
              Write('>> Learning rate (fast) = ', LearningRate: 9: 7, ' with 0..30: 0.01; 31..100: 0.005; 101..400: 0.001; 401..800: 0.0005; else 0.0001.');
            RolledOffLearning:
              // Learning Rolled off learning rate.
              Write('>> Learning rate (rolled of) = ', LearningRate: 9: 7, ' Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
          end;
        end
        else
        // Display override learning rate.
        Write('>> Learning rate (override) = ', LearningRate: 9: 7, '.');

        // Display training information.
        Writeln(' Window # = ', WindowCount, ' Weight decay = ', WeightDecay: 8: 6, '; Decay per epoch = ', 100.0 * (1.0 - Power(DecayScale, WindowCount)): 0: 6,
          '%; Temperature = ', TTemperature: 8: 6, '; Clip limit = ', ClipLimit: 8: 6, '.');
      end;
    end;      // End epoch loop.
  except
    on E: Exception do begin
      TrainSuccess := False;
      Writeln('Error in training: ', E.ClassName, ': ', E.Message);
      Pause;
      Exit;
    end;
  end;

  Writeln;
  Writeln('Training ended.');
  TrainSuccess := True;
  Pause;
end;

end.

