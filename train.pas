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

// Compute the CE loss.
function ComputeCELoss(const Probs: TSeqVocabMatrix; const TargetTokens: TIDimVector): Double;
const
  Eps = 1.0e-12;
var
  i, Tok: Integer;
  P: Double;
begin
  Result := 0.0;

  for i := 0 to SeqLen - 1 do begin
    Tok := TargetTokens[i];
    P := Probs[i, Tok];

    if P < Eps then
      P := Eps;

    Result := Result - Ln(P);
  end;

  Result := Result / SeqLen;
end;

// Compute the CE loss only.
function ComputeLossOnly(var WModelParams: TWModelParams; var WModelState: TWModelState; const InputTokens, TargetTokens: TIDimVector; const Start: Integer): Double;
var
  Blk, LastBlk: Integer;
begin
  with WModelParams do with WModelState do begin

    LaunchEmbeddingLookup(Embeddings.dValue, dInputTokens, StateBlock[0].X.dValue, SeqLen, ModelDim);
    CuScale(CuHandle, SeqLen * ModelDim, Scale, StateBlock[0].X.dValue);

    for Blk := 0 to nBlock - 1 do begin
      RunTransformForward(WModelParams, WModelState, Blk, Start);

      if Blk < nBlock - 1 then
        cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);
    end;

    LastBlk := nBlock - 1;
    CuMatMulFullNT(CuHandle, StateBlock[LastBlk].X7.dValue, Embeddings.dValue, dProbs, SeqLen, nVocab, ModelDim, ModelDim, ModelDim, DimVocab);
    LaunchSoftmaxForwardStrided(dProbs, dProbs, SeqLen, nVocab, DimVocab, Temperature);
    cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
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

// Run the training.
procedure RunTrain(var WModelParams: TWModelParams; var WModelState: TWModelState; const TokenizedCorpus: TIVector);
const
  Sqrt10 = 3.16227766016837933199;
var
  i, j, k: Integer;
  Blk, LastBlk, Epoch, Start, WindowCount, MinLossEpoch: Integer;
  Loss, MinLoss, DiffLoss, LastLoss, EpochLoss, MEL, StartLoss,
    MeanRunningLoss, RecentLossSum, GradScale: Double;
  RecentLosses: array[0..RecentCount] of Double;
  RecentLossIndex, RecentLossCount: Integer;
  // PreUpdateLoss, PostUpdateLoss: Double; // Not using currently.

  function TrainReadIfKeyPressed: Boolean;
  var
    key: char;
    ModelFileName: string;
  begin
    Result := False;
    key := CheckForControlKey;
    case key of
      'p', 'P': begin        // Pause work.
        Write('Pause requested. Hit <CR> to continue.');
        Readln;
        Result := False;
      end;
      'x', 'X': begin        // Exit training. Success, go to main menu.
        Writeln('Exit requested. Stopping training.');
        TrainSuccess := False;
        Result := True;
      end;
      'n', 'N': begin        // Exit training. No success, go to inference.
        Writeln('Stopping training.');
        TrainSuccess := True;
        Result := True;
      end;
      'v', 'V': begin        // Enable verbose transform.
        VerboseTransform := not VerboseTransform;
        Writeln('Very verbose transform mode: ', VerboseTransform);
        Pause;
      end;
      'i', 'I': begin        // Report program info.
        Writeln;
        ReportInfo;
        Pause;
      end;
      't', 'T': begin        // Display training info.
        Writeln('Training. nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' ModelDim = ', ModelDim, ' Projection = ', Proj,
          '  Epoch = ', Epoch, ' Start = ', Start, ' Stride = ', Stride, ' Length of tokens in corpus = ', Length(TokenizedCorpus));
        Writeln(DateTimeToStr(Now), '  X = Exit training. P = Pause. N = Go to iNference. V = toggle Verbose mode.  I = program Information. T = Training information. S = Save. Training...');
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
  // For running losses.
  RecentLossIndex := 0;
  RecentLossCount := 0;
  RecentLossSum := 0.0;
  LastLoss := 0.0;
  for i := 0 to 9 do
    RecentLosses[i] := 0.0;

  // For each epoch's loss.
  MinLoss := 1000000;
  MinLossEpoch := -1;

  // Initialization.
  StopTraining := False;
  Training := False;               // Set False for debugging.
  GlobalSeed := 123456789;         // For debugging.
  //GlobalSeed := GetTickCount64;  // For training.
  if NewModel then
    nVocab := nSymbols;            // Need nVocab (second name for variable) for Transform.

  // Check DimVocab is large enough.
  if nVocab > DimVocab then begin
    Writeln('nVocab > DimVocab. Aborting training....');
    TrainSuccess := False;
    Exit;
  end;

  // Initialize state.
  InitializeTransformerState(WModelState);

  // Initialize params if new model.
  if NewModel then
    InitializeTransformerParams(WModelParams);

  // Initiate Cuda.
  StartCuda(WModelParams, WModelState);

  try
    CopyParamsToDevice(WModelParams);
    CopyInvFreqToDevice(WModelState);

    // Initialize epoch/sequence loop.
    Epoch := 0;
    Start := 0;
    GlobalStep := 0;
    Writeln('Start Training');
    Writeln('X = Exit training. P = Pause. V = toggle Verbose mode. I = program Information. T = Training information. S = Save. Transforming...');

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
      Start := (Epoch * 17) mod Stride;

      // Stride loop thru Sequence.
      while ((Start + SeqLen + 1) <= Length(TokenizedCorpus)) do begin

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
          for i := 0 to SeqLen - 1 do
            Write(i:4, ' token=', InputTokens[i], ' ');
          Writeln('Values for row zero.');
          k := InputTokens[0];
          for j := 0 to 15 do
            Write(WModelParams.Embeddings.Value[k, j]:8:5, ' ');
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

        // Forward pass thru transformer.
        for Blk := 0 to nBlock - 1 do begin

          RunTransformForward(WModelParams, WModelState, Blk, Start);

          if Blk < nBlock - 1 then
            // CopyXTensor(StateBlock[Blk].X7, StateBlock[Blk + 1].X);
            cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);

        end;

        LastBlk := nBlock - 1;

        // 3. FORWARD HEAD OUTPUT STAGE.

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
            // Equation: Embeddingsᵀ.Grad = X7ᵀ · TopGradient. Embeddingsᵀ.Grad in R^{nVocab x D}. X7ᵀ in R^(D x L}. TopGradient in R^{L x nVocab}.
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
        end; // End gradient stage.

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
        LaunchClipVector(WModelParams.Embeddings.dGrad, DimVocab * ModelDim, ClipLimit);
        CheckCudaError('Gradient clipping.');

        // Modify learning rate and decay scale.
        // Remember Training may be False (affects dropouts).
        // Remember GlobalStep := WindowCount * Epoch; GS is redundant.
        // Weight decay tied to learning rate, AdamW style.
        Case Epoch of
          0..100: LearningRate := 0.01;
          101..200: LearningRate := 1.0 /(100 * Sqrt10);
          201..500: LearningRate := 0.001;
          501..1000: LearningRate := 1.0 / (1000 * Sqrt10);
         else LearningRate := 0.00005;
        end;

        // User can set override.
        if OverrideLearningRate <> -1.0 then
          LearningRate := OverrideLearningRate;

        // LearningRate := FloorLearningRate + (BaseLearningRate - FloorLearningRate) * Power(Rolloff, GlobalStep);
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
        CheckFirstValuesFinite('Embeddings', WModelParams.Embeddings.dValue, nVocab * ModelDim);
        for k := 0 to nBlock - 1 do with WModelParams.ParamBlock[k] do begin
          CheckFirstValuesFinite('Wq', Wq.dValue, ModelDim * ModelDim);
          CheckFirstValuesFinite('W1', W1.dValue, ModelDim * ModelDimProj);
          CheckFirstValuesFinite('W2', W2.dValue, ModelDimProj * ModelDim);
          CheckFirstValuesFinite('Gamma1', Gamma1.dValue, ModelDim);
          CheckFirstValuesFinite('Gamma2', Gamma2.dValue, ModelDim);
        end;

        // Update sequence loop.
        Start := Start + Stride;
        EpochLoss := EpochLoss + Loss;
        Inc(WindowCount);
        Inc(GlobalStep);
      end;    // End sequence loop.

      // Compute mean and minimum loss.
      if WindowCount > 0 then
        MEL := EpochLoss / WindowCount
      else
        MEL := 0.0;
      if MEL < MinLoss then begin
        MinLoss := MEL;
        MinLossEpoch := Epoch;
      end;

      // Difference from previous epoch. Positive DiffLoss means improvement.
      if Epoch = 0 then
        DiffLoss := 0.0
      else
        DiffLoss := LastLoss - MEL;
      LastLoss := MEL;

      // Rolling mean over last RecentCount (typically 10) epochs.
      if RecentLossCount < RecentCount then begin
        Inc(RecentLossCount);
      end else begin
        RecentLossSum := RecentLossSum - RecentLosses[RecentLossIndex];
      end;
      RecentLosses[RecentLossIndex] := MEL;
      RecentLossSum := RecentLossSum + MEL;
      RecentLossIndex := (RecentLossIndex + 1) mod RecentCount;
      MeanRunningLoss := RecentLossSum / RecentLossCount;

      if StopTraining then Exit;

      // Display loss progress.
      if Epoch = 0 then
        StartLoss := MEL;
      if (Epoch mod 20) = 0 then begin
        Writeln('>> nTC = ', Length(TokenizedCorpus), ' nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' Stride = ', Stride,
          ' ModelDim = ', ModelDim, ' nHead = ', nHead, ' nBlock = ', nBlock, ' Proj = ', Proj, ' DropOut = ', not Training);
        Writeln('>> Learning rate = ', LearningRate:10:8, ' Floor LR = ', FloorLearningRate:10:8, ' Base LR = ', BaseLearningRate:10:8,
          ' LR rolloff = ', RollOff:10:8, ' Weight decay = ', WeightDecay:10:8, ' Decay scale = ', DecayScale:10:8);
        Writeln('>> Training = ', Training, ' Temperature = ', Temperature: 10: 8, ' Clip limit = ', ClipLimit: 10: 8, ' Global step = ', GlobalStep);
      end;
      if MinLossEpoch = Epoch then
        Write('^^')
      else
        Write('--');
      Write('Epoch ', Epoch, ' has ended. Window count = ', WindowCount, '. Mean loss: Start = ', StartLoss:10:8, '; Minimum = ',
        MinLoss:10:8, ' in epoch ', MinLossEpoch, '; Current = ', MEL:10:8);
      Write('; Rolling', RecentLossCount, ' = ', MeanRunningLoss:10:8);

      if Epoch > 0 then begin
        if DiffLoss > 0 then
          Write('; Better by ', DiffLoss:10:8)
        else
          Write('; Worse by ', -DiffLoss:10:8);
      end;

      Writeln(' in epoch ', Epoch, '.');
    end;      // End epoch loop.
  except
    Writeln('Error in training.');
    Pause;
  end;        // End try loop.

  Writeln;
  Writeln('End of training.');
  TrainSuccess := True;
  Pause;
end;

end.

