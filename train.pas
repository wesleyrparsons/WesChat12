unit Train;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

interface

uses
  Display,
  Global,
  IOHandler,
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
  Scale = Sqrt(ModelDim);         // Optional transformer-style embedding scaling by sqrt(d_model).

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
var
  i, j, k: Integer;
  Blk, LastBlk, Epoch, Start, WindowCount: Integer;
  Loss, EpochLoss: Double;
  GradScale: Single;

  function TrainReadIfKeyPressed: Boolean;
  var
    key: char;
    ModelFileName: string;
  begin
    Result := False;
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Result := True;
      end;
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        Result := True;
      end;                   // Break out of the loop cleanly.
      'p', 'P': begin
        Write('Pause requested. Hit any key to continue.');
        Readln;
        Result := False;
      end;                   // Break out of the loop cleanly.
      'v', 'V': begin
        VeryVerboseTransform := not VeryVerboseTransform;
        Writeln('Very verbose transform mode: ', VeryVerboseTransform);
        Pause;
      end;                   // Change verbosity.
      'i', 'I': begin
        Writeln;
        ReportInfo;          // Report program info.
        Pause;
      end;
      't', 'T': begin
        Writeln('Training. nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' ModelDim = ', ModelDim, ' Projection = ', Proj,
          '  Epoch = ', Epoch, ' Start = ', Start, ' Stride = ', Stride, ' Length of tokens in corpus = ', Length(TokenizedCorpus));
        Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of loop. P = Pause. V = toggle Very Verbose mode.');
        Writeln('I = program Information. T = Training information. S = Save. Training...');
        Pause;
      end;
      's', 'S': begin
        ChDir(WorkingDir);   // Save model.
        Write('Enter filename: ');
        Readln(ModelFileName);
        if SaveModel(ModelFileName, WModelParams) then
          Writeln('File ', ModelFileName, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
      end;

    end;
  end;

begin
  StopTraining := False;
  Training := True;               // Set False for debugging.
  GlobalSeed := 123456789;        // For debugging.
  //GlobalSeed := GetTickCount64;   // For training.
  nVocab := nSymbols;             // Need nVocab (second name for variable) for Transform.

  // Start training.
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

  StartCuda(WModelParams, WModelState);
{  // Initiate Cublas.
  InitializeCuBLAS;

  // Allocate cuda memory.
  if not CudaAllocated then
    MAllocCublas(WModelParams, WModelState);
  CheckCudaError('Train -- initialize cuda.');}

  try
    CopyParamsToDevice(WModelParams);
    CopyInvFreqToDevice(WModelState);

    // Initialize epoch/sequence loop.
    Epoch := 0;
    Start := 0;
    Writeln('** Start training. nVocab = ', nVocab, ' DimVocab = ', DimVocab, ' Seqlen = ', SeqLen, ' Epochs = ', MaxEpochs, ' ModelDim = ', ModelDim, ' Projection = ', Proj);
    Writeln('** X = Exit program. B = Break out of loop. P = Pause. V = toggle Very Verbose mode,');
    Writeln('** I = program Information. T = Training information. S = Save. Transforming...');

    // Display embeddings.
    Writeln('Display Embeddings.Value prior to Transform.');
    DisplayX(WModelParams.Embeddings.Value, B);
    Pause;

    // Loop through epochs.
    for Epoch := 0 to MaxEpochs - 1 do with WModelState do begin
      EpochLoss := 0;
      WindowCount := 0;
      Start := (Epoch * 17) mod Stride;

      // Stride loop thru Sequence.
      while ((Start + SeqLen + 1) <= Length(TokenizedCorpus)) and (not StopTraining) do begin

        // Display number of loops thru embed loop.
        if DisplayWindow then
          Writeln('Epoch is ', Epoch, ' SeqLen window is from ', Start, ' to ', Start + SeqLen, ' on loop ', WindowCount);

        // Build the target vector, one ahead, for the loss stage.
        BuildTargetVector(TargetTokens, TokenizedCorpus, Start, SeqLen);
        BuildInputVector(InputTokens, TokenizedCorpus, Start, SeqLen);
        cudaMemcpy(dInputTokens, @InputTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);
        cudaMemcpy(dTargetTokens, @TargetTokens[0], SeqLen * SizeOf(Integer), cudaMemcpyHostToDevice);

        // Checking.
        if VeryVerboseTransform then begin
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
        if VeryVerboseTransform then begin
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
          if StopTraining then Break;

          RunTransformForward(WModelParams, WModelState, Blk, Start);

          if Blk < nBlock - 1 then
            // CopyXTensor(StateBlock[Blk].X7, StateBlock[Blk + 1].X);
            cudaMemcpy(StateBlock[Blk + 1].X.dValue, StateBlock[Blk].X7.dValue, XSize, cudaMemcpyDeviceToDevice);

          if PauseIfKeyPressed then
            StopTraining := TrainReadIfKeyPressed;
        end;

        LastBlk := nBlock - 1;
        if StopTraining then Break;

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
          if VeryVerboseTransform then begin
            cudaMemcpy(@Probs[0, 0], dProbs, ProbsSize, cudaMemcpyDeviceToHost);
            VTPDisplayX('Display Probs, in transform, before softmax.', Probs, B);
           end;

          // Display Embeddings.Value matrix.
          if VeryVerboseTransform then begin
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
          Loss := ComputeCELoss(Probs, TargetTokens);  //```
          if VeryVerboseTransform then
            VTPDisplayX('Display Probs, in transform, after softmax.', Probs, B);

          // 3C. Cross-Entropy Loss. Obtain TopGradient from Probs.
          if DisplaySubstage then Writeln('': Stage, '3C. Transform Forward, Obtain TopGradient from Probs');
          // Gradient: Input Probs. Output TopGradient. Also option of CalculateGradient from KLDivergence.
          // Equation: TopGradient in R^{L x nVocab}. Probs in R^{L x nVocab}.
          // GradientFromCEProbabilities(WModelState);  // Using CE.
          // GradientFromKLDivergence(WModelState);   // Not using KL.
          // LaunchCEGradient(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab); Replaced with strided version.
          LaunchCEGradientStrided(dProbs, dTopGradient, dTargetTokens, SeqLen, nVocab, DimVocab);

          // Diagnostic helper. ``
          GradScale := 1.0 / SeqLen;
          CuScale(CuHandle, ProbsSize div SizeOf(Single), GradScale, dTopGradient);

          // Display TopGradient matrix.
          if VeryVerboseTransform then begin
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
            if VeryVerboseTransform then begin
              cudaMemcpy(@X7.Grad[0, 0], X7.dGrad, XSize, cudaMemcpyDeviceToHost);
              VTPDisplayX('Display X7.Grad, in transform, after stage 2D.', X7.Grad, G);
            end;
          end;
        end; // End gradient stage.

        if VerboseTransform then
          Writeln('            Switch from Forward to Backprop');

        // Backprop pass thru transformer.
        for Blk := nBlock - 1 downto 0 do with WModelState do begin
          if StopTraining then Break;

          RunTransformBackprop(WModelParams, WModelState, Blk);

          if Blk > 0 then
            cudaMemcpy(StateBlock[Blk - 1].X7.dGrad, StateBlock[Blk].X.dGrad, XSize, cudaMemcpyDeviceToDevice);

          if PauseIfKeyPressed then
            StopTraining := TrainReadIfKeyPressed;
        end;

        // Modify weights and biases.
        for k := 0 to nBlock - 1 do
          Optimization(WModelParams, k);

        // Apply the total embedding gradient (output-side + input-side).
        UpdateEmbeddings(wModelParams, WModelState);

        // Diagnostic check for excessive embeddings values.
        CheckFirstValuesFinite('Embeddings', WModelParams.Embeddings.dValue, nVocab * ModelDim);
        for k := 0 to nBlock - 1 do begin
          CheckFirstValuesFinite('Wq', WModelParams.ParamBlock[k].Wq.dValue, ModelDim * ModelDim);
          CheckFirstValuesFinite('W1', WModelParams.ParamBlock[k].W1.dValue, ModelDim * ModelDimProj);
          CheckFirstValuesFinite('W2', WModelParams.ParamBlock[k].W2.dValue, ModelDimProj * ModelDim);
          CheckFirstValuesFinite('Gamma1', WModelParams.ParamBlock[k].Gamma1.dValue, ModelDim);
          CheckFirstValuesFinite('Gamma2', WModelParams.ParamBlock[k].Gamma2.dValue, ModelDim);
        end;

        Start := Start + Stride;
        EpochLoss := EpochLoss + Loss;
        Inc(WindowCount);

      end;    // End sequence loop.
      Write('Epoch ', Epoch, ' has ended. Window count = ', WindowCount, ' Epoch mean loss = ');
      if WindowCount > 0 then
        Writeln((EpochLoss / WindowCount): 8: 6)
      else
        Writeln('0.000000');

    end;      // End epoch loop.
  finally
  // Clean up cuda and cublas.
{    if CudaAllocated then
      MDeallocateCublas(WModelParams, WModelState);
    if CuBLAS_Shutdown then
      Writeln('CuBLAS successfully shut down.')}
  end;

  Writeln;
  Writeln('End of training.');
  TrainSuccess := True;
  Pause;
end;

end.

