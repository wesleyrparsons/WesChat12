unit Embed;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.1, January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Math,
  SysUtils,
  Transform;

 {From the tokenization stage:
  TokenizedCorpus are vector of Integers, which become InputTokens and TargetTokens.
  Arrays are nSymbols x ModelDim of Single.
  nSymbols is vocabulary size. ModelDim is the dimension of the models, the loads.}

procedure RunEmbed(const TokenizedCorpus: TIVector);
procedure TestEmbedding;

implementation

const
  MaxSeq = 128;                        // Need maximum size of sequence to dimension array.
  Scale = Sqrt(ModelDim);

var
  Embeddings: array of array of Single;        // Row is token, column is weights.
  Block: Integer;

// Test emnbedding with small model of 4 tokens.
procedure TestEmbedding;
const
  TestVocab = 5;
  TestSeq = 4;
var
  i, j: Integer;
  Tokens: array[0..TestSeq - 1] of Integer; // For testing, tokens.
  XTest: TSeqMatrix;                        // For tesing, sequence.
  ETest: TEmbeddingMatrix;                  // For testing, embedding matrix.
begin
  Writeln('--- Embedding Test ---');

  { 1. Define a small token sequence. }
  Tokens[0] := 0;
  Tokens[1] := 1;
  Tokens[2] := 2;
  Tokens[3] := 3;

  { 2. Create deterministic embeddings. }
  for i := 0 to TestVocab - 1 do
    for j := 0 to ModelDim - 1 do
      ETest[i, j] := i + 0.01 * j;  // Easy to verify visually.

  { 3.  Standardizatio to keep in range. }
  for i := 0 to SeqLen - 1 do
    for j := 0 to ModelDim - 1 do
      X[i, j] := X[i, j] * Scale;

  { 4. Display result. }
  for i := 0 to TestSeq - 1 do begin
    Write('Token ', Tokens[i], ': ');
    for j := 0 to Min(7, ModelDim - 1) do
      Write(X[i, j]:8:4, ' ');
    Writeln;
  end;

  Writeln('--- End Test ---');
  Readln;
end;

// Create the target vector for use in head output.
procedure BuildTargetVector(var Target: TIDimVector; const TokenizedCorpus: TIVector;
  const StartIndex, L: Integer);
var
  i: Integer;
begin
  // Target must be sized [0..L - 1].
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
      X[i, j] := Embeddings[id, j];
  end;
end;

// Run the training.                          Make this a Raw Byte Vector.
procedure RunEmbed(const TokenizedCorpus: TIVector);
var
  i, j, k: Integer;
  Start, EmbedLoop: Integer;
  Stride: Integer = 64;

  procedure ReadEmbedIfKeyPressed;
  var
    key: char;
  begin
    key := CheckForControlKey;
    case key of
      'x', 'X': begin
          writeln('Exit requested. Stopping execution.');
          Pause;
          Halt;              // Immediately terminate program.
        end;
      'b', 'B': begin
          writeln('Break requested. Exiting loop.');
          Pause;
          Block := nBlock;   // Break out of the loop cleanly.
        end;
      'v', 'V': begin
        VeryVerbose := not VeryVerbose;
        writeln('Very verbose mode: ', VeryVerbose);
        Pause;
      end;                   // Change verbosity.
      'p', 'P': begin
        writeln;
        ReportInfo;          // Report program info.
        Pause;
      end;
      'e', 'E': begin
        writeln('Embedding. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim + 1,
          '  Start = ', Start, ' Stride = ', Stride, ' SeqLen = ', SeqLen, ' MaxSeq = ', MaxSeq, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));
        writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
        writeln(' P = Program information. E = Embedding information. Embedding & transforming...');
        Pause;
      end;
    end;
  end;

begin
  if VeryVerbose then begin
    writeln('Start Embedding. nVocab = ', nVocab, ' nSymbols = ', nSymbols, ' ModelDim = ', ModelDim + 1,
       ' SeqLen = ', SeqLen, ' MaxSeq = ', MaxSeq, ' Length of TokenizedCorpus = ', Length(TokenizedCorpus));
  end;

  // Set the dimensions of the embedding matrix.
  SetLength(Embeddings, nSymbols);
  for i := 0 to nSymbols - 1 do
    SetLength(Embeddings[i], ModelDim);

  // Seed the weights with random numbers.
  for i := 0 to nSymbols - 1 do             // Random normal distribution.
    for j := 0 to ModelDim - 1 do           // Mean = 0, SD = 0.02.
      Embeddings[i, j] := RandG(0.0, 0.02); // Only time I use this randomizer.

  writeln('First quarter of two rows of embeddings.');
  for k := 0 to ModelDim div 4 - 1 do
    write(Embeddings[1, k]: 8: 6, ' ');
  writeln;
  for k := 0 to ModelDim div 4 - 1 do
    write(Embeddings[2, k]: 8: 6, ' ');
  writeln;
  Pause;

  // Initialize.
  InitializeTransformer;

  // Stride loop theu Sequence.
  Start := 0;
  EmbedLoop := 0;
  while (Start + SeqLen) < Length(TokenizedCorpus) do begin

    // Display number of loops thru embed loop.
    Inc(EmbedLoop);
    writeln('&&& Loop thru Embed: start ', Start, ' and loop number ', EmbedLoop, ' &&&');
    writeln(DateTimeToStr(Now), '  X = Exit program. B = Break out of merge loop. V = toggle Verbose mode.');
    writeln('  P = Program information. E = Embedding information. Embedding & transforming...');

    if VerboseTransform then Pause;

    // Build X from TokenizedCorpus[start .. start + SeqLen - 1].
    BuildInputMatrix(X, TokenizedCorpus, Start, SeqLen);

    // Standardizatio to keep in range.
    for i := 0 to SeqLen - 1 do
      for j := 0 to ModelDim - 1 do
        X[i, j] := X[i, j] * Scale;

    // Build the target vector, one ahead, for the loss stage.
    BuildTargetVector(TargetTokens, TokenizedCorpus, Start + 1, SeqLen);

    if VerboseTokenize then begin
      writeln('Display X, beginning, after PE, before transform.');
      DisplayX(X, G);
      Pause;
    end;

    // Forward and backward pass thru transformer.
    for Block := 0 to nBlock - 1 do begin
      writeln('$$$ Starting Block ', Block, '  Sequence Start ', Start, ' $$$');
      if VerboseTransform then Pause;

      RunTransform;

      if PauseIfKeyPressed then
        ReadEmbedIfKeyPressed;

      Start := Start + Stride;
    end;
  end;

  nVocab := nSymbols;    // I have two names for this variable.
  writeln('End of training. Press <CR> to continue.');
  Readln;
end;

end.


