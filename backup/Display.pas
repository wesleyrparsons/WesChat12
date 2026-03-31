unit Display;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global;

// Pause procedures.
procedure HardPause;
procedure Pause;

// Interrupt procedures.
function CheckForControlKey: Char;

// Display symbols procedures.
function CleanUpSymbol(const x: RawByteString): RawByteString;
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);

// Display vectors and matrices.
procedure DisplayVector(const V: TIVector);
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TVocabWeightMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TVocabWeightMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;

// Report information on program.
procedure ReportInfo;

implementation

uses
  Crt,
  DateUtils,
  Math,
  SysUtils;

// Pause, unconditional.
procedure HardPause;
begin
  Write('Hit <CR> to continue.... ');
  Readln;
  Writeln;
end;

// Pause, subject to DoNotPause.
procedure Pause;
var
  tt: TDateTime;
begin
  if not DoNotPause then begin
    tt := Now;
    HardPause;
    Stoptime := StopTime + Now - tt;
  end;
end;

// Returns key pressed.
function CheckForControlKey: Char;
begin
  if KeyPressed then
    Result := ReadKey
  else
    Result := #0;   { means: no key }
end;

// Write information on state of program.
procedure ReportInfo;
begin
  Writeln('--- Program Information ---');
  Writeln('WesChat, Version: ', Version);
  Writeln('Author: Wesley R. Parsons');
  Writeln('Date: begun January 10, 2026');
  Writeln('Sequence Length (SeqLen): ', SeqLen);
  Writeln('Model Dimensions (ModelDim): ', ModelDim);
  Writeln('Dimensional Projections (Proj): ', Proj);
  Writeln('Heads (nHead): ', nHead);
  Writeln('Blocks (nBlock): ', nBlock);
  Writeln('Learning Rate (LearningRate): ', LearningRate: 6: 4);
  Writeln('Trainable Parameters: Wq, Wk, Wv, W0, W1, b1, W2, b2, gamma1, beta1, beta2, gamma2');
  Writeln('Maximum Vocabulary (MaxVocab): ', MaxVocab);
  Writeln('Number of Vocabulary (nVocab): ', nVocab);
end;

// Replace unprintable symbols with space.
function CleanUpSymbol(const x: RawByteString): RawByteString;
var
  j, L: Integer;
  ch: Char;
begin
  L := Length(x);
  SetLength(Result, L);   // Allocate output string

  for j := 1 to L do begin
    ch := x[j];

    if Ord(ch) in [1..31, 127..255] then
      Result[j] := ' '
    else
      Result[j] := ch;
  end;
end;

// Display the symbol table.
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
var
  i: Integer;
begin
  Writeln('--- Symbol Table ---');
  for i := 0 to High(SymbolTable) do begin  // Loop thru each symbol in table.
    if (i in [0..31]) or (i in [127..255]) then
      Write(i: 8, IntToHex(i, 2): 15)       // Hex for non-ASCII characters.
    else
      Write(i: 8, '"' + SymbolTable[i] + '"': 15);
    if (i mod 5) = 4 then Writeln;
    if (i > 0) and (i mod 100 = 99) then Pause;
  end;
  Writeln;
  Writeln('Symbol table length = ', Length(SymbolTable));
  Writeln;
end;

// Display a vector, character by character, then pause.
procedure DisplayVector(const V: TIVector);
var
  i: Integer;
begin
  for i := 0 to High(V) do   // V is an array of integers.
    Write(V[i], ' ');
  Writeln;
  Pause;
end;

// Display scope of display below.
procedure PartScope(const Part: TPart);
begin
  Case Part of
    B: Writeln(' Beginning.');
    E: Writeln(' End.');
    F: Writeln(' Full.');
    G: Writeln(' Sample.');
  end;
end;

// Display an X matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Floor(Length(X) / tStride);
      hStride := Floor(Length(X[0]) / tStride);
      iB := 0;
      iE := tStride;
      jB := 0;
      jE := tStride;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Hidden matrix, B, E, F, or G.
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Floor(Length(X) / tStride);
      hStride := Floor(Length(X[0]) / tStride);
      iB := 0;
      iE := tStride;
      jB := 0;
      jE := tStride;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: THiddenMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Vocab Weight matrix, B, E, F, or G.
procedure DisplayX(const X: TVocabWeightMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Floor(Length(X) / tStride);
      hStride := Floor(Length(X[0]) / tStride);
      iB := 0;
      iE := tStride;
      jB := 0;
      jE := tStride;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 5, ' ');
    Write(X[High(X), High(X[0])]: 11: 5, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TVocabWeightMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a Vocab TSeq matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Floor(Length(X) / tStride);
      hStride := Floor(Length(X[0]) / tStride);
      iB := 0;
      iE := tStride;
      jB := 0;
      jE := tStride;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TSeqVocabMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display a ScoresHead matrix, B, E, F, or G.
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
const
  tStride = 10;
var
  i, j, iB, iE, jB, jE: Integer;
  vStride: Integer = 1;
  hStride: Integer = 1;
begin
  Case Part of
    B: begin
      iB := 0;
      iE := 9;
      jB := 0;
      jE := 9;
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
    end;
    F: begin
      iB := 0;
      iE := High(X);
      jB := 0;
      jE := High(X[0]);
    end;
    G: begin
      vStride := Floor(Length(X) / tStride);
      hStride := Floor(Length(X[0]) / tStride);
      iB := 0;
      iE := tStride;
      jB := 0;
      jE := tStride;
    end;
  end;
  Write('       ');
  for j := jB to jE do
    Write(j * hStride: 8, '    ');
  if Part = G then
    Write(High(X[0]): 8, '    ');
  Writeln;
  for i := iB to iE do begin
    Write(i * vStride: 4);
    for j := jB to jE do
      Write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      Write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
  if Part = G then begin
    Write(High(X): 4);
    for j := jB to jE do
      Write(X[High(X), j * hStride]: 11: 7, ' ');
    Write(X[High(X), High(X[0])]: 11: 7, ' ');
    Writeln;
  end;
end;

// Conditional form of DisplayX.
procedure VTPDisplayX(const Mess: string; const X: TScoresMatrix; const Part: TPart = B); overload;
begin
  if VerboseTransform then begin
    Write(Mess);
    PartScope(Part);
    DisplayX(X, Part);
    Pause;
  end;
end;

// Display for ScoresHead1 and 2, 0..20, 0..15.
procedure DisplayScoresHead(const ScoresHead: TScoresMatrix);
var
  i, j:Integer;
begin
  for i := 0 to 20 do
    for j := 0 to 15 do
      Write(ScoresHead[i, j]: 11: 6);
  Writeln;
  Pause;
end;

end.
