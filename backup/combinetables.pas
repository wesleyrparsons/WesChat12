unit CombineTables;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  SysUtils;

type
  TSymbolTable = array of string;

procedure MergeSymbolTables(var CombinedTable: TSymbolTable);

implementation

var
  BOS, EOS, PAD, UNK: Integer;                   // Extra symbols for control.

{ Helper: compare for descending length, then alphabetical }
function CompareForGreedy(const a, b: string): Integer;
begin
  Result := Length(b) - Length(a);               { longer first }
  if Result = 0 then
    Result := CompareStr(a, b);                  { stable alpha tie-breaker }
end;

procedure LoadOneSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  Magic: array[0..3] of Char;
  i, Len: Integer;
  S: string;
begin
  Assign(F, FileName);
  Reset(F, 1);

  // Magic header.
  BlockRead(F, Magic, SizeOf(Magic));
  if (Magic[0] <> 'S') or (Magic[1] <> 'Y') or
     (Magic[2] <> 'M') or (Magic[3] <> 'T') then begin
    Close(F);
    writeln('Invalid symbol table file.');
    Exit;
  end;

  // Version.
  BlockRead(F, Version, 16);

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, NSymbols);

  // Special token IDs
  BlockRead(F, BOS, SizeOf(BOS));
  BlockRead(F, EOS, SizeOf(EOS));
  BlockRead(F, PAD, SizeOf(PAD));
  BlockRead(F, UNK, SizeOf(UNK));

  // Read symbols.
  for i := 0 to nSymbols - 1 do begin
    BlockRead(F, Len, SizeOf(Len));
    SetLength(S, Len);
    if Len > 0 then
      BlockRead(F, S[1], Len);
    SymbolTable[i] := S;
  end;

  Close(F);
  nSymbols := Length(SymbolTable);
  nVocab := nSymbols;
  Writeln('Loaded ', nSymbols, ' symbols from ', FileName);
end;

{ Merge any number of symbol tables into one:
   - removes duplicates
   - sorts for greedy L->R (longest match first) }
procedure MergeSymbolTables(var CombinedTable: TSymbolTable);
var
  i, j, k, Total, Count: Integer;
  s, Temp, Line, ListFile: string;
  Found: Boolean;
  F: TextFile;
  FilesRead: TSVector;
  Tables: array of TSymbolTable;
begin

  //LoadOneSymbolTable('Nam0.sym', Tables[0]);
  //LoadOneSymbolTable('Nam1.sym', Tables[1]);

  write('Enter name of file list: ');
    readln(ListFile);
    if not FileExists(ListFile) then begin
      Writeln('List file not found: ', ListFile);
      Pause;
      Exit;
    end;

    AssignFile(F, ListFile);
    Reset(F);

    Count := 0;
    SetLength(FilesRead, 0);

  while not EOF(F) do begin
    ReadLn(F, Line);
    Line := Trim(Line);
    if Line = '' then
      Continue;         // Skip blank lines.
    if FileExists(Line) then begin
      LoadOneSymbolTable(Line, Tables[Count]);
      Writeln('  File processed: ', Line, '; symbol bytes read: ', Length(Tables[Count]));

      Inc(Count);
      SetLength(FilesRead, Count);
      FilesRead[Count - 1] := Line;
    end
    else begin
      Writeln('  File not found: ', Line);
      Pause;
    end;
  end;

  CloseFile(F);

  { 1. Rough capacity }
  Total := 0;
  for i := Low(Tables) to High(Tables) do
    Inc(Total, Length(Tables[i]));

  SetLength(CombinedTable, Total);
  k := 0;

  { 2. Union (deduplicate) }
  for i := Low(Tables) to High(Tables) do
    for j := 0 to High(Tables[i]) do begin
      s := Tables[i, j];
      Found := False;
      for Total := 0 to k - 1 do
        if CombinedTable[Total] = s then begin
          Found := True;
          Break;
        end;
      if not Found then begin
        CombinedTable[k] := s;
        Inc(k);
      end;
    end;

  SetLength(CombinedTable, k);

  { 3. Sort once for deterministic greedy L->R }
  for i := 0 to High(CombinedTable) do
    for j := i + 1 to High(CombinedTable) do
      if CompareForGreedy(CombinedTable[i], CombinedTable[j]) > 0 then begin
        Temp := CombinedTable[i];
        CombinedTable[i] := CombinedTable[j];
        CombinedTable[j] := Temp;
      end;
end;

end.


