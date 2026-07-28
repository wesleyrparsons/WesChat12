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
var
  LastBestSaveEpoch: Integer;     // Saving models.
  AutoSaveEpoch: Integer;         // At this epoch, start saving new minimum loss models.
  BestSavedLoss: Double;
  MinSaveGap: Integer;
  MinSaveDelta: Double;

// Compute the CE loss.
function ComputeCELoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;
const
  Eps = 1.0e-12;
var
  i, Tok: Integer;
  P, RawP: Double;
  TinyCount: Integer;
begin
  Result := 0.0;
  TinyCount := 0;

  for i := 0 to SeqLen - 1 do begin
    Tok := TargetTokens[i];

    // Check token before indexing Probs.
    if (Tok < 0) or (Tok >= nVocab) then
      raise Exception.CreateFmt('ComputeCELoss: invalid target token %d at position %d; nVocab=%d', [Tok, i, nVocab]);
    P := Probs[i, Tok];

    // Check NaN/Inf before doing comparisons.
    if IsNan(P) then begin
      P := Eps;
      Writeln('NaN probability at position ', i, ', token ', Tok, 'decoded as ', Decode(Tok), '. Using P = ', Eps: 9: 7, '. Pausing...');
      Pause;
    end;

    if IsInfinite(P) then begin
      P := 0.99999999;
      Writeln('Infinite probability at position ', i, ', token ', Tok, 'decoded as ', Decode(Tok), '. Using P = 0.99999999.');
      Pause;
    end;

    if P < 0.0 then begin
      P := Eps;
      Writeln('Probability less than 0.0, ', P: 9: 7, ', at position ', i, ', token ', Tok, ' decoded as ', Decode(Tok), '. Using P = ', P: 9: 7, '. Pausing...');
      Pause;
    end;

    if P > 1.0 then begin
      P := 0.99999999;
      Writeln('Probability greater than 1.0, ', P: 9: 7, ', at position ', i, ', token ', Tok, ' decoded as ', Decode(Tok), '. Using P = ', P: 9: 7, '. Pausing...');
      Pause;
    end;

    if P < Eps then begin
      RawP := P;
      Inc(TinyCount);
      if TinyCount >= SeqLen - 2 then begin
        Write('Tiny probabilities below Eps in CE loss. Count = ', TinyCount, ' out of ', SeqLen);
        Writeln('. Sequence position = ', i,', target token  = ', Tok, ' decoded as ', Decode(Tok), '. Raw P = ', RawP: 9: 7, '. Using P = ', Eps: 9: 7, '.');
      end;
      P := Eps;
    end;

    Result := Result - Ln(P);
  end;

  Result := Result / SeqLen;

  if TinyCount > 48 then begin
    Writeln('Warning: ', TinyCount, ' probabilities were below epsilon of ', Eps: 9: 7, ' in this CE loss.  Pausing...');
    Pause;
  end;
end;

// Compute the CE loss only.
function ComputeLossOnly(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens, TargetTokens: TIDimVector): Double;
var
  Blk: Integer;
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

    RunOutputForward(WModelParams, WModelState);
    cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
    if DebugCudaChecks then
      CheckCudaError('Copy probabilities for diagnostic loss.');

    Result := ComputeCELoss(Probs, TargetTokens);
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
  Blk, Epoch, Start, WindowCount, MinLossEpoch,
    RecentLossIndex, RecentLossCount: Integer;
  Loss, MinLoss, DiffLoss, LastLoss, EpochLoss, MEL, StartLoss,
    MeanRunningLoss, RecentLossSum: Double;
  RecentLosses: array[0..RecentCount - 1] of Double;
  Starts: TIVector;
  NeedParamCopy: Boolean;
  // PreUpdateLoss, PostUpdateLoss: Double; // Not using currently.

  function TrainReadIfKeyPressed: Boolean;
  var
    key: char;
    ModelFileName: string;
  begin
    Result := False;
    key := CheckForControlKey;
    case key of
      'x', 'X': begin        // Exit training. Success, go to main menu.
        Writeln('Exit requested. Stopping training.');
        TrainSuccess := False;
        Result := True;
        StopTraining := True;
      end;
      'n', 'N': begin        // Exit training. No success, go to inference.
        Writeln('Stopping training. Going to inference...');
        TrainSuccess := True;
        Result := True;
        StopTraining := True;
      end;
      'v', 'V': begin        // Enable verbose transform.
        VerboseTransform := not VerboseTransform;
        Writeln('Very verbose transform mode: ', VerboseTransform);
      end;
      'w', 'W': begin        // Report program info.
        Writeln;
        ReportProgramInfo;
        Pause;
      end;
      'p', 'P':              // Pause work.
        Pause;
      'l', 'L': begin        // Override learning rate.
        Write('Enter override learning rate: ');
        Readln(OverrideLearningRate);
      end;
      'i', 'I': begin        // Display training info.
        Writeln('Training. Work = ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), '; nTC = ', Length(TokenizedCorpus), ',  nVocab = ', nVocab, '; DimVocab = ', DimVocab,
          '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock, '; Proj = ', Proj, '; DropOut = ', Training, '.');
        Write(DateTimeToStr(Now), '  X = Exit training. N = go to iNference. V = toggle Verbose mode.  ');
        Writeln('I = program Information. P = Pause. L = set Learning rate. S = Save. Training...');
        Pause;
      end;
      's', 'S': begin        // Save model.
        Write('Enter model filename, blank for automatic checkpoint: ');
        Readln(ModelFileName);

        ModelFileName := Trim(ModelFileName);

        if ModelFileName = '' then begin
          ModelFileName := ModelDir + WorkingName + '_epoch' + IntToStr(Epoch) +
            '_step' + IntToStr(GlobalStep) + '_' + FormatDateTime('yyyy-mm-dd_hhnnss', Now) + '.model';
        end
        else begin
          // If user typed only a bare filename, save it in ModelDir.
          if ExtractFilePath(ModelFileName) = '' then
            ModelFileName := ModelDir + ModelFileName;

          // Add extension if missing.
          if ExtractFileExt(ModelFileName) = '' then
            ModelFileName := ModelFileName + '.model';
        end;

        if SaveModel(ModelFileName, WModelParams) then
          Writeln('File ', ModelFileName, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
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

  // For running losses.
  RecentLossIndex := 0;
  RecentLossCount := 0;
  RecentLossSum := 0.0;
  LastLoss := 0.0;
  for i := 0 to High(RecentLosses) do
    RecentLosses[i] := 0.0;

  // For each epoch's loss.
  MEL := 0;
  MinLoss := 1000000;
  MinLossEpoch := -1;

  // Initialization.
  StopTraining := False;           // Control var.
  Training := True;
  GlobalSeed := 123456789;         // For debugging.
  //GlobalSeed := GetTickCount64;  // For training.
  if NewModel then
    nVocab := nSymbols;            // Need nVocab (second name for variable) for Transform.

  // Check DimVocab is large enough.
  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting training...');
    TrainSuccess := False;
    Exit;
  end;

  // Initialize state.
  InitializeTransformerState(WModelState);

  // Initialize params if new model.
  if NewModel then
    InitializeTransformerParams(WModelParams);

  NeedParamCopy := (not CudaAllocated) or ParamsNeedCopyToDevice or NewModel;

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
    Epoch := 0;
    Start := 0;
    GlobalStep := 0;
    Writeln('Training started.');
    Write(DateTimeToStr(Now), '  X = Exit training. N = go to iNference. V = toggle Verbose mode.  ');
    Writeln('I = program Information. P = Pause. L = set Learning rate. S = Save. Training...');

    // Display embeddings.
    if VerboseTransform then begin
      Writeln('Display Embeddings.Value prior to Transform.');
      DisplayX(WModelParams.Embeddings.Value, B);
      Pause;
    end;

    // Loop through epochs.
    for Epoch := 0 to MaxEpochs - 1 do with WModelState do begin
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

        {// 3. FORWARD HEAD OUTPUT STAGE.

        with WModelParams do with WModelState do begin
          Stage := Blk * 3 + 2;

          // 3A. Multiplication/Overwrite. Obtain Probs from X7 and Vocab.
          if DisplayStage then Writeln('' : Stage, 'Stage  3, Block ', Blk, ',  Transform Forward');
          if DisplaySubstage then Writeln('': Stage, '3A. Transform Gradient, Obtain Probs from X7 and Vocab');

          // Multiplication: Input X7, Vocab. Output Probs.
          // Equation: Probs = X7 · Embeddingsᵀ. Probs in R^{L x nVocab}. X in R^{L x D}.  Embeddings in R^{nVocab x D}.
          // MatMulFullNT(@StateBlock[LastBlk].X7.Value[0, 0], @Embeddings.Value[0, 0], @Probs[0, 0], SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);
          CuMatMulFullNT(CuHandle, StateBlock[LastBlk].X7.dValue, Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);

          // Display Probs matrix.
          if VerboseTransform then begin
            cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
            VTPDisplayX('Display Probs, in transform, before softmax.', Probs, B);
           end;

          // Display Embeddings.Value matrix.
          if VerboseTransform then begin
            cudaMemcpy(@Embeddings.Value[0, 0], Embeddings.dValue, EmbeddingsSize, cudaMemcpyDeviceToHost);
            VTPDisplayX('Display Embeddings.Value in transform, before computing Logit.', Embeddings.Value, B);
          end;

          // 3B. Softmax. Obtain Probs from Sotmax(Probs).
          if DisplaySubstage then Writeln('': Stage, '3B. Transform Forward, Softmax Probs');

          // Softmax: Input Probs. Output Probs.
          // Equation: Probs = Softmax(Probs).
          // for i := 0 to SeqLen - 1 do
          //   SoftmaxForwardN(@Probs[i,0], @Probs[i,0], nVocab);
          // LaunchSoftmaxForwardN(dProbs, dProbs, SeqLen, nVocab, Temperature);
          LaunchSoftmaxForwardStrided(dProbs, dProbs, SeqLen, nVocab, DimVocab, Temperature);

          // Display Probs matrix.
          cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);

          // Compute loss.
          Loss := ComputeCELoss(Probs, TargetTokens);
          // PreUpdateLoss := Loss;

          if VerboseTransform then
            VTPDisplayX('Display Probs, in transform, after softmax.', Probs, B);

          // 3C. Cross-Entropy Loss. Obtain TopGradient from Probs.
          if DisplaySubstage then Writeln('': Stage, '3C. Transform Forward, Obtain TopGradient from Probs');
          // Gradient: Input Probs. Output TopGradient. Also option of CalculateGradient from KLDivergence.
          // Equation: TopGradient in R^{L x nVocab}. Probs in R^{L x nVocab}.
          // GradientFromCEProbabilities(WModelState);  // Using CE.
          // GradientFromKLDivergence(WModelState);   // Not using KL.
          // LaunchCEGradient(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab); Replaced with strided version.
          LaunchCEGradientStrided(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab, DimVocab);

          // Scale gradients.
          GradScale := 1.0 / SeqLen;
          // GradScale := 1.0;  Diagnostic helper.
          CuScale(CuHandle, ProbsSize div SizeOf(Single), GradScale, dTopGradient);

          // Display TopGradient matrix.
          if VerboseTransform then begin
            cudaMemcpy(@TopGradient[0, 0], dTopGradient, ProbsSize, cudaMemcpyDeviceToHost);
            VTPDisplayX('Display TopGradient, in transform, after Logit calculation.', TopGradient, B);
          end;

          // 3D. Backprop TopGradient creates X7 Grad: Input TopGradient, WVocabᵀ. Output X7.Grad.
          if DisplaySubstage then Writeln('': Stage, '3D. Transform Backprop, Create X7 from TopGradient');

          with StateBlock[LastBlk] do begin
            // Equation: X7.Grad = TopGradient · Embeddings.Value. X7.Grad in R^{L x D}. TopGradient in R^{L x nVocab}. Embeddings.Value in R^{nVocab x D}.
            // MatMulFullNN(@TopGradient[0, 0], @Embeddings.Value[0, 0], @X7.Grad[0, 0], SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);
            {cblas_sgemm(101, 111, 111, SeqLen, ModelDim, nVocab, 1.0, @TopGradient[0, 0], DimVocab, @Embeddings.Value[0, 0], ModelDim, 0.0, @X7.Grad[0, 0], ModelDim);}
            CuMatMulFullNN(CuHandle, dTopGradient, Embeddings.dValue, X7.dGrad, SeqLen, ModelDim, nVocab, DimVocab, ModelDim, ModelDim);

            // Backprop TopGradient modifies/overwrites Embeddingsᵀ: Input X7ᵀ, TopGradient. Output Embeddingsᵀ.Grad.
            // Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L). TopGradient in R^{L x nVocab}.
            // MatMulFullAccTN(@TopGradient[0,0], @X7.Value[0,0], @Embeddings.Grad[0,0], nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);
            CuMatMulFullAccTN(CuHandle, dTopGradient, X7.dValue, Embeddings.dGrad, nVocab, ModelDim, SeqLen, DimVocab, ModelDim, ModelDim);

            // GradSplit(X7.Grad, X5.Grad, X6.Grad, SeqLen, ModelDim);
            // CuGradSplit(CuHandle, X7.dGrad, X5.dGrad, X6.dGrad, SeqLen, ModelDim); Moved to Backprop.

            // Display X7.Grad matrix.
            if VerboseTransform then begin
              cudaMemcpy(@X7.Grad[0, 0], X7.dGrad, XSize, cudaMemcpyDeviceToHost);
              VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', X7.Grad, G);
            end;
          end;
        end; // End gradient stage.}

        // Run the output-head forward pass.
        // This produces probabilities in dProbs.
        RunOutputForward(WModelParams, WModelState);

        // Training loss uses every sequence position, so copy complete probability matrix to the host.
        cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
        if DebugCudaChecks then
          CheckCudaError('Copy probabilities for training loss.');

        // Compute mean cross-entropy loss for the current window.
        Loss := ComputeCELoss(Probs, TargetTokens);

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

        // Use schedule to calculate learning rate. Remember Training may be False (affects dropouts).
        Case LearningStyle of
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

        // Diagnostic check for pre- versus post-update losses.
        {PostUpdateLoss := ComputeLossOnly(WModelParams, WModelState, InputTokens, TargetTokens, Start);
        Writeln('Pre update loss  = ', PreUpdateLoss:10:8);
        Writeln('Post update loss = ', PostUpdateLoss:10:8);
        Pause;}

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

      // Compute minimum epoch loss.
      if WindowCount = 0 then begin
        Writeln('Epoch ', Epoch, ' contained no training windows.');
        Continue;
      end;

      MEL := EpochLoss / WindowCount;

      // Difference from previous epoch. Positive DiffLoss means improvement.
      if Epoch = 0 then
        DiffLoss := 0.0
      else
        DiffLoss := LastLoss - MEL;
      LastLoss := MEL;

      // Rolling mean over last RecentCount (typically 10) epochs.
      if RecentLossCount < RecentCount then
        Inc(RecentLossCount)
      else
        RecentLossSum := RecentLossSum - RecentLosses[RecentLossIndex];
      RecentLosses[RecentLossIndex] := MEL;
      RecentLossSum := RecentLossSum + MEL;
      RecentLossIndex := (RecentLossIndex + 1) mod RecentCount;
      MeanRunningLoss := RecentLossSum / RecentLossCount;

      if StopTraining then Exit;

      // Display loss progress.
      if Epoch = 0 then
        StartLoss := MEL;
      if (Epoch mod 10) = 1 then begin

        // Parameters.
        Write('>> Work = ', ExtractFileName(ExcludeTrailingPathDelimiter(WorkingDir)), '; nTC = ', Length(TokenizedCorpus), '; nVocab = ', nVocab, '; DimVocab = ', DimVocab,
          '; Seqlen = ', SeqLen, '; Stride = ', Stride, '; ModelDim = ', ModelDim, '; nHead = ', nHead, '; nBlock = ', nBlock, '; Proj = ', Proj, '; DropOut = ', Training);
        if Training then
          Writeln(' (', ADropOut: 4: 3, ' ', MLPDropOut: 4: 3, ' ', RDropOut: 4: 3, ').')
        else
          Writeln('.');

        if OverrideLearningRate = -1.0 then begin
          // Not override learning rate.
          Case LearningStyle of
            SlowLearning:
              // Display slow learning rate schedule.
              Write('>> Learning rate (slow) = ', LearningRate: 9: 7, ' with 0..10: 0.01; 11..20: 0.005; 21..100: 0.0005; 101..1000: 0.0001; else 0.00005. ');
            FastLearning:
              // Display fast learning rate schedule.
              Write('>> Learning rate (fast) = ', LearningRate: 9: 7, ' with 0..30: 0.01; 31..100: 0.005; 101..400: 401..800: .0005; else 0.0001. ');
            RolledOffLearning:
              // Learning Rolled off learning rate.
              Write('>> Learning rate (rolled of) = ', LearningRate: 9: 7, ' Floor LR = ', FloorLearningRate: 9: 7, ' Base LR = ', BaseLearningRate: 9: 7, ' LR rolloff = ', RollOff: 9: 7, '.');
          end;
        end
        else
        // Display override learning rate.
        Write('>> Learning rate (override) = ', LearningRate: 9: 7, '. ');

        Writeln('Weight decay = ', WeightDecay: 9: 7, '; Decay scale = ', DecayScale: 9: 7, '; Perplexity = ', exp(MEL): 9: 7,'.');
        Writeln('>> Temperature = ', TTemperature: 9: 7, '; Clip limit = ', ClipLimit: 9: 7, '; Global step = ', GlobalStep, '.');
      end;

      // Save best model subject to conditions.
      if MEL < MinLoss then begin
        MinLoss := MEL;
        MinLossEpoch := Epoch;
        // Writeln('New minimum loss in Epoch ', Epoch, '. MinLoss = ', MinLoss: 9: 7);

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
      Write('Epoch ', Epoch, ' ended. LR = ', LearningRate: 10 :8, '. Window count = ', WindowCount, '. Mean loss: Start = ', StartLoss: 9: 7, '; Minimum = ',
        MinLoss: 9: 7, ' in epoch ', MinLossEpoch, '; Current = ', MEL: 9: 7);
      Write('; Rolling', RecentLossCount, ' = ', MeanRunningLoss: 9: 7);

      if Epoch > 0 then begin
        if DiffLoss > 0 then
          Write('; Better by ', DiffLoss: 9: 7)
        else
          Write('; Worse by ', -DiffLoss: 9: 7);
      end;
      Writeln('.');

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

