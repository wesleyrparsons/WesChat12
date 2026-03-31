unit IOHandler;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Classes,
  Crt,
  DateUtils,
  Display,
  FileUtil,
  Global;

procedure ReadFileBytes(const FileName: string; var OneCorpus: TBVector);
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);

implementation

var
  BOS: Integer = 256;
  EOS: Integer = 257;
  PAD: Integer = 258;
  UNK: Integer = 259;
  Magic: array[0..3] of Char = ('S', 'Y', 'M', 'T');  // For saving symbol table.

procedure ReadFileBytes(const FileName: String; var OneCorpus: TBVector);
var
  F: File;
  Size, i: Integer;
  B: Byte;
begin
  AssignFile(F, FileName);
  Reset(F, 1);     // Open in binary mode.
  Size := FileSize(F);
  SetLength(OneCorpus, Size);

  // Write the Corpus as it is read.
  if VeryVerbose and VerboseTokenize then
    Writeln('--- Original Corpus ---');
  for i := 0 to Size - 1 do begin
    BlockRead(F, B, 1);
    OneCorpus[i] := B;

    if VeryVerbose and VerboseTokenize then
      if ShowEachByteRead then
        if B < 32 then
          Write('<', B, '>')
        else
          Write(Chr(B));
  end;
  CloseFile(F);
  if VeryVerbose and VerboseTokenize then begin
    Writeln('ReadByteFile: ');
    for i := 0 to 150 do
      Write(OneCorpus[i], ' ');
    Readln;
  end;
  if VeryVerbose and VerboseTokenize then
    Writeln;

  // Display initial Corpus length.
  Writeln('Read ', Size, ' bytes from ', FileName);
end;

// Load the symbol table from file. IOHandler.
procedure LoadSymbolTable(const FileName: string; var SymbolTable: TSymbolTable);
var
  F: file;
  Magic: array[0..3] of Char;
  S: string;
  i, Len: Integer;
begin
  BOS := 256;
  EOS := 257;
  PAD := 258;
  UNK := 259;
  Assign(F, FileName);
  Reset(F, 1);

  // Magic header.
  BlockRead(F, Magic, SizeOf(Magic));
  if (Magic[0] <> 'S') or (Magic[1] <> 'Y') or
     (Magic[2] <> 'M') or (Magic[3] <> 'T') then begin
    Close(F);
    Writeln('Invalid symbol table file.');
    Pause;
    Exit;
  end;

  // Version.
  BlockRead(F, Version, 16);

  // Symbol count.
  BlockRead(F, nSymbols, SizeOf(nSymbols));
  SetLength(SymbolTable, NSymbols);

  // Special token IDs.
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
  Writeln('Loaded ', nSymbols, ' symbols from ', FileName);
end;

// Save symbol table.
procedure SaveSymbolTable(const SymbolFileName: string; const SymbolTable: TSymbolTable);
var
  F: file;
  NumSymbols: Integer;
  i, Len: Integer;
begin
  Assign(F, SymbolFileName);
  ReWrite(F, 1);

  // Magic.
  BlockWrite(F, Magic, SizeOf(Magic));

  // Version.
  BlockWrite(F, Version, 16);

  // Symbol count.
  NumSymbols := Length(SymbolTable);
  BlockWrite(F, NumSymbols, SizeOf(NumSymbols));

  // Special token IDs.
  BlockWrite(F, BOS, SizeOf(BOS));
  BlockWrite(F, EOS, SizeOf(EOS));
  BlockWrite(F, PAD, SizeOf(PAD));
  BlockWrite(F, UNK, SizeOf(UNK));

  // Write each symbol.
  for i := 0 to NumSymbols - 1 do begin
    Len := Length(SymbolTable[i]);
    BlockWrite(F, Len, SizeOf(Len));
    if Len > 0 then
      BlockWrite(F, SymbolTable[i][1], Len);
  end;

  Close(F);
  Writeln('File ', SymbolFileName, ' successfully saved.');
end;

// Load tokenized corpus from a token file.
procedure LoadTokenList(const TokenFileName: string; var TokenizedCorpus: TIVector);
var
  F: file of Integer;
  v, i, Count: Integer;
begin
  AssignFile(F, TokenFileName);
  Reset(F);

  // Determine number of tokens in file.
  Count := FileSize(F);

  // Allocate TokenizedCorpus.
  SetLength(TokenizedCorpus, Count);

  // Read them back.
  for i := 0 to Count - 1 do begin
    Read(F, v);
    TokenizedCorpus[i] := v;
  end;

  CloseFile(F);
  nTokenizedCorpus := Length(TokenizedCorpus);
  Writeln('Loaded ', Count, ' tokens from ', TokenFileName);
end;

// Save the output tokenized corpus to a token file.
procedure SaveTokenList(const TokenizedCorpus: TIVector; const TokenFileName: String);
var
  F: file of Integer;
  v, i: Integer;
begin
  AssignFile(F, TokenFileName);
  ReWrite(F);

  for i := 0 to High(TokenizedCorpus) do begin
    v := TokenizedCorpus[i];
    Write(F, v);
  end;

  CloseFile(F);
  Writeln('File ', TokenFileName, ' successfully saved.');
  Writeln;
end;

end.
