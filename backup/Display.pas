unit Display;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global;

procedure HardPause;
procedure Pause;
function CheckForControlKey: Char;
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
procedure DisplayVector(const V: TIVector);
procedure DisplayX(const X: TSeqMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TSeqHeadMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: THiddenMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TSeqVocabMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TVocabWeightMatrix; const Part: TPart = B); overload;
procedure DisplayX(const X: TScoresMatrix; const Part: TPart = B); overload;
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
  write('Hit <CR> to continue.... ');
  Readln;
  writeln;
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

// Pause, and waits for key press.
procedure xPauseProcIfKeyPressed;
begin
  writeln;
  Pause;   // Waits for the key.
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
  writeln('--- Program Information ---');
  writeln('WesChat, Version: ', Version);
  writeln('Author: Wesley R. Parsons');
  writeln('Date: February 15, 2026');
  writeln('Sequence Length (SeqLen): ', SeqLen);
  writeln('Model Dimensions (ModelDim): ', ModelDim);
  writeln('Dimensional Projections (Proj): ', Proj);
  writeln('Heads (nHead): ', nHead);
  writeln('Blocks (nBlock): ', nBlock);
  writeln('Learning Rate (LearningRate): ', LearningRate: 6: 4);
  writeln('Trainable Parameters: Wq, Wk, Wv, W0, W1, b1, W2, b2, gamma1, beta1, beta2, gamma2');
  writeln('Maximum Vocabulary (MaxVocab): ', MaxVocab);
  writeln('Number of Vocabulary (nVocab): ', nVocab);
end;

// Display the symbol table.
procedure DisplayByteSymbolTable(const SymbolTable: TSymbolTable);
var
  i: Integer;
begin
  Writeln('--- Symbol Table ---');
  for i := 0 to High(SymbolTable) do begin  // Loop thru each symbol in table.
{    case i of
      7, 8, 9, 10, 11, 12, 13, 127:
        write(i: 8, ' ': 15)     // Placeholder for dangerous characters.
      else
        write(i: 8, '"' + SymbolTable[i] + '"': 15);
    end;}
    if (i < 32) or (i > 126) then
      write(i: 8, IntToHex(i, 2): 15)       // Hex for non-ASCII characters.
    else
      write(i: 8, '"' + SymbolTable[i] + '"': 15);
    if (i mod 5) = 4 then writeln;
    if (i > 0) and (i mod 100 = 99) then Pause;
  end;
  writeln('Symbol table length = ', Length(SymbolTable));
  writeln;
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 5, ' ');
    write(X[High(X), High(X[0])]: 11: 5, ' ');
    writeln;
  end;
end;

// Display an XHead matrix, B, E, F, or G.
procedure DisplayX(const X: TSeqHeadMatrix; const Part: TPart = B); overload;
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 5, ' ');
    write(X[High(X), High(X[0])]: 11: 5, ' ');
    writeln;
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 5, ' ');
    write(X[High(X), High(X[0])]: 11: 5, ' ');
    writeln;
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 5, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 5, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 5, ' ');
    write(X[High(X), High(X[0])]: 11: 5, ' ');
    writeln;
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 7, ' ');
    write(X[High(X), High(X[0])]: 11: 7, ' ');
    writeln;
  end;
end;

// Display a Scores matrix, B, E, F, or G.
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
      // if High(X[0]) < 9 then jE := High(X[0]);
    end;
    E: begin
      iB := High(X) - 9;
      iE := High(X);
      jB := High(X[0]) - 9;;
      jE := High(X[0]);
      // if High(X[0]) < 9 then jE := High(X[0]);
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
  write('       ');
  for j := jB to jE do
    write(j * hStride: 8, '    ');
  if Part = G then
    write(High(X[0]): 8, '    ');
  writeln;
  for i := iB to iE do begin
    write(i * vStride: 4);
    for j := jB to jE do
      write(X[i * vStride, j * hStride]: 11: 7, ' ');
    if Part = G then
      write(X[i * vStride, High(X[0])]: 11: 7, ' ');
    writeln;
  end;
  if Part = G then begin
    write(High(X): 4);
    for j := jB to jE do
      write(X[High(X), j * hStride]: 11: 7, ' ');
    write(X[High(X), High(X[0])]: 11: 7, ' ');
    writeln;
  end;
end;

// Display for ScoresHead1 and 2, 0..20, 0..15.
procedure DisplayScoresHead(const ScoresHead: TScoresMatrix);
var
  i, j:Integer;
begin
  for i := 0 to 20 do begin
    for j := 0 to 15 do
      write(ScoresHead[i, j]: 11: 6);
  writeln;
  end;
  Pause;
end;

end.
