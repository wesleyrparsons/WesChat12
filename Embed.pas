unit Embed;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  IOHandler,
  Math,
  SysUtils,
  Transform,
  Util;

 {TokenizedCorpus is a vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols (nVocab) is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunEmbed(const TokenizedCorpus: TIVector);

implementation

const
  Scale = Sqrt(ModelDim);    // Optional transformer-style embedding scaling by sqrt(d_model).

var
  WModel: WModelType;        // WModel is declared here. (Change to WModel.)
  Block: Integer;            // Number of iterations sequentially of Transform.

// Create the target vector for use in head output.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector;
  const StartIndex, L: Integer);
var
  i: Integer;
begin
  Assert(StartIndex >= 0);
  Assert(StartIndex + L <= Length(TokenizedCorpus));

  for i := 0 to L - 1 do
    Target[i] := TokenizedCorpus[StartIndex + i];
end;

// Create the input matrix.
procedure BuildInputMatrix(var X: TSeqMatrix; const TokenizedCorpus: TIVector;
  const Start, L: Integer);
var
  i, j, id: Integer;
begin
  // Bounds check on the corpus window.
  Assert(Start >= 0);
  Assert(Start + L <= Length(TokenizedCorpus));

  for i := 0 to L - 1 do begin
    id := TokenizedCorpus[Start + i];

    // Validate token ID.
    Assert(id >= 0);
    Assert(id < nSymbols);

    // Copy embedding vector.
    for j := 0 to ModelDim - 1 do
      X[i, j] := WModel.Embeddings.Value[id, j];
  end;
end;

// Run the training.
procedure RunEmbed(const TokenizedCorpus: TIVector);
var
  i, j, k: Integer;
  Start, EmbedLoop: Integer;
  Stride: Integer = 64;      // Stride 64 tokens every sequence.

  procedure ReadEmbedIfKeyPressed;
  var
    key: char;
    Success: Boolean;
    ModelFileName: string;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
        Writeln('Exit requested. Stopping execution.');
        Pause;
        Halt;                // Immediately terminate program.
      end;
      'b', 'B': begin
        Writeln('Break requested. Exiting loop.');
        Pause;
        Block := nBlock;     // Break out of the loop cleanly.
      end;
      'v', 'V': begin
        VeryVerbose := not VeryVerbose;
        Writeln('Very verbose mode: ', VeryVerbose);
        Pause;
      end;                   // Change verbosity.
      'i', 'I': begin
        Writeln;
        ReportInfo;          // Report program info.
        Pause;
      end;
      't', 'T': begin
        Writeln('Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
          '  Start = ', Start, ' Stride = ', Stride, ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));
        Write(DateTimeToStr(Now), '  X = Exit program. B = Break out of loop. V = toggle Verbose mode. P = Pause.');
        Writeln('  W = WesChat Information. T = Training information. S = Save. Training...');
        Pause;
      end;
      's', 'S': begin
        ChDir(WorkingDir);   // Save model.
        Write('Enter filename: ');
        Readln(ModelFileName);
        SaveModel(ModelFileName, WModel, Success);
        ChDir('..');
        if Success then
          Writeln('File ', f, ' successfully saved.')
        else
          Writeln('File not saved.');
        Pause;
      end;

    end;
  end;

begin
  nVocab := nSymbols;    // Need nVocab (second name for variable) for Transform.

  if VeryVerbose then
    Writeln('Start Training. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim,
      ' SeqLen = ', SeqLen, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));

  // Seed the weights with random numbers.
  for i := 0 to nSymbols - 1 do             // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      WModel.Embeddings.Value[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  Writeln('First quarter of two rows of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModel.Embeddings.Value[1, k]: 8: 6, ' ');
  Writeln;
  for k := 0 to ModelDim div 4 - 1 do
    Write(WModel.Embeddings.Value[2, k]: 8: 6, ' ');
  Writeln;
  Pause;

  VTPDisplayX('Display Embeddings.Value prior to Transform.', WModel.Embeddings.Value, B);

  // Initialize.
  InitializeTransformer(WModel);
  SetLength(TokenID, Length(TokenizedCorpus));
  TokenID := TokenizedCorpus;

  // Stride loop thru Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(TokenizedCorpus) do begin

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    Writeln('&&& Loop thru Embed: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    Writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    Writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
    BuildInputMatrix(X.Value, TokenizedCorpus, Start, SeqLen);

    // Optional transformer-style embedding scaling by sqrt(d_model).
    for i := 0 to SeqLen - 1 do
      for j := 0 to ModelDim - 1 do
        X.Value[i, j] := X.Value[i, j] * Scale;

    // Build the target vector, one ahead, for the loss stage.
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start + 1, SeqLen);

    VTPDisplayX('Display X.Value before transform.', X.Value, G);

    // Forward and backward pass thru transformer.
    for Block := 0 to nBlock - 1 do begin
      Writeln('$$$ Starting Block ', Block, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransform(WModel);

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;
    end;

    Start := Start + Stride;
  end;

  Writeln('End of training. Press <CR> to continue.');
  Readln;
end;

end.

