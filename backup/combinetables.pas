unit CombineTables;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  IOHandler,
  SysUtils;

type
  TSymbolTable = array of string;

procedure MergeSymbolTables(var CombinedTable: TSymbolTable);

implementation

{ Helper: compare for descending length, then alphabetical }
function CompareForGreedy(const a, b: string): Integer;
begin
  Result := Length(b) - Length(a);               { longer first }
  if Result = 0 then
    Result := CompareStr(a, b);                  { stable alpha tie-breaker }
end;

{ Merge any number of symbol tables into one:  - removes duplicates
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
      LoadSymbolTable(Line, Tables[Count]);
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


