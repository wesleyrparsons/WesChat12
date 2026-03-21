program WesChat;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

uses
  Crt,
  GPT2Tokenize,
  Display,
  Embed,
  FileUtil,
  Global,
  IOHandler,
  Transform,
  Symbolize,
  SysUtils,
  Tokenize,
  WesTokenize,
  Windows,
  CombineTables;

var
  Ch, CorpusFileName, SymbolFileName, MergeFileName, ListFile: string;
  OldLen: Integer;
  Corpus: TBVector;           // Vector of byte.
  TokenizedCorpus: TIVector;  // Make this one big dimensioned vector at start.
  // Allocate size of corpus, make nTC count the size.

// Read a file of file names, and sends each to tokenizer.
procedure ProcessFileList(var ListFile: string; Corpus: TBVector);
var
  F: TextFile;
  Line: string;
  FilesRead: TSVector;
  CombinedCorpus: TBVector;
  i, Count: Integer;
begin
  MultipleCorpus := True;
  MultipleFileName := EmptyStr;
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
  FromSymbolTable := False;

  while not EOF(F) do begin
    ReadLn(F, Line);
    Line := Trim(Line);
    if Line = '' then Continue;         // Skip blank lines.
    if FileExists(Line) then begin
      if Count = 0 then
        WorkingName := ChangeFileExt(CorpusFileName + 'Mult', '');
      ReadFileBytes(Line, Corpus);
      SetLength(CorpusFileNames, Count);
      CorpusFileNames[Count] := Line;
      Writeln('  File processed: ', Line, '; corpus bytes read: ', Length(Corpus));

      OldLen := Length(CombinedCorpus);
      SetLength(CombinedCorpus, OldLen + Length(Corpus));

      for i := 0 to High(Corpus) do
        CombinedCorpus[OldLen + i] := Corpus[i];
      Writeln('Total bytes read: ', Length(CombinedCorpus));
      Inc(Count);
      SetLength(FilesRead, Count);
      FilesRead[Count - 1] := Line;
    end
    else begin
      Writeln('  File not found: ', Line, '.');
      Pause;
    end;
  end;

  CloseFile(F);
  SetLength(Corpus, Length(CombinedCorpus));
  for i := 0 to High(CombinedCorpus) do
    Corpus[i] := CombinedCorpus[i];

  writeln('Combined corpus length = ', Length(Corpus));
  nCorpus := Length(Corpus);
  Pause;
  RunTokenize(Corpus);
  HardPause;

end;

procedure Help;
begin
  writeln('  H: Help');
  writeln('  VTO: VerboseTokenize := True');
  writeln('  VV: VeryVerbose := True');
  writeln('  VTR: VerboseTransform := True');
  writeln('  NVTO: VerboseTokenize := False');
  writeln('  NVV: VeryVerbose := False');
  writeln('  NVTR: VerboseTransform := False');
  writeln('  DNP: DoNotPause := True');
  writeln('  DP: DoNotPause := False');
  writeln('  SF: SaveFiles := True');
  writeln('  NSF: SaveFiles := False');
  writeln('  M: Maximum merges: ');
  writeln('  PC: Maximum pair count: ');
  writeln('  LR: Learning rate: ');
  writeln;
end;

begin
  { Necessary because JSON will throw dupe errors otherwise }
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);
  { Below is not working on my Lazarus console }
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  //WriteLn('Testing UTF‑8: äöü ß é ñ 中 文 😀'); Pause;
  writeln('WesChat, Version 1.2, begun January 19, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.');
  writeln;
  writeln('Options:');
  writeln('  1: Tokenize an input single file using WesChat''s byte-level byte-pair encoding, with');
  writeln('     deterministic left-to-right longest-prefix matching and greedy longest-match decoding.');
  writeln('     Write the symbol table and other information to disk.');
  writeln('  2: Tokenize using WesChat an input set of files listed one per line in a file.');
  writeln('     The concatenated token list will appear in the first file.');
  writeln('  3: Tokenize input Bela corpus using input Bela symbol table.');
  writeln('  4: Tokenize an input single file, based on an input symbol table, ');
  writeln('     using WesChat''s tokenizer.');
  writeln('  5: Tokenize Bela corpus using ChatGPT''s symbol and merge tables and WesChat''s');
  writeln('     tokenization routine.');
  writeln('  6: Tokenize an input single file using ChatGPT''s symbol and merge tables and WesChat''s');
  writeln('     tokenization routine.');
  writeln('  7: Tokenize an input single file using input symbol and merge tables.');
  writeln('  8: Combine two symbol tables for use with WesChat tokenization.');
  writeln('  9: Create symbol table from input corpus.');
  writeln('  X: Exit.');
  writeln;
  writeln('Ater tokenization, WesChat will begin training the transformer, which consists');
  writeln('of 4 to 8 blocks. The attention stage has 8 heads. There are a weight stage wih a bias');
  writeln('and a weight stage without a bias. The activation function is softmax with temperature.');
  writeln('Model dimensions are 160 or 256. The activation stage expands dimensionality fourfold.');
  writeln('Precision is single. Sequence length is 128 or 256 bytes. Pre-layer normalization');
  writeln('standardizes for means and standard deviations. Attention and residual dropouts are 0.1.');
  writeln('All output files will be contained in a folder or file named with the input file name,');
  writeln('appended with a timestamp.');
  // writeln;
  while True do begin
    write('> ');
    readln(Ch);
    Case UpperCase(Ch) of
      '1': begin
        // Ask user for input file.
        Write('Enter input filename: ');
        Readln(CorpusFileName);

        // Read bytes from file.
        if FileExists(CorpusFileName) then begin
          ReadFileBytes(CorpusFileName, Corpus);
          SetLength(CorpusFileNames, 1);
          CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
           + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

          FromSymbolTable := False;
          WorkingName := ChangeFileExt(CorpusFileName, '');
          RunTokenize(Corpus);
          if nSymbols > 0 then
            RunEmbed(TokenizedCorpus)
          else
            writeln('Symbols not found in table.');
        end
        else
          writeln('File not found: ', FileName, '.');
      end;
      '2': ProcessFileList(ListFile, Corpus);
      '3': begin
        FromSymbolTable := True;
        if FileExists('bela.sym') then begin
            SymbolFileName := 'bela.sym';
            SetLength(CorpusFileNames, 1);     // may not be necessary for multiple input corpuses.
            CorpusFileNames[0] := SymbolFileName;
            LoadSymbolTable(SymbolFileName, SymbolTable);
            ReadFileBytes('bela.txt', Corpus);
            RunWesTokenize(Corpus, SymbolTable)
          end
        else writeln('File not found');
      end;
      '4': begin
        // Ask user for input file.
        write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;  // Delete this var at some point.

        if not FileExists(SymbolFileName) then
          Writeln('Symbol table file not found: ', SymbolFileName, '. Aborting...')
        else begin
          LoadSymbolTable(SymbolFileName, SymbolTable);
          if Length(SymbolTable) < 2 then
            writeln('Too few symbols (< 2) found. Aborting...')
          else begin
            write('Input Corpus file name: ');
            Readln(CorpusFileName);

            if not FileExists(CorpusFileName) then
              Writeln('Corpus file not found: ', CorpusFileName, '. Aborting...')
            else begin
              ReadFileBytes(CorpusFileName, Corpus);
              WorkingName := ChangeFileExt(CorpusFileName, '');
              SetLength(CorpusFileNames, 1);     // may not be necessary for multiple input corpuses.
              CorpusFileNames[0] := CorpusFileName;

              // Use WesTokenize here.
              RunWesTokenize(Corpus, SymbolTable);
{              if VerboseTokenize then
                WriteTokenList(B);
              DisplaySymbolTable(SymbolTable);
              Pause;
              DetokenizeToDisplay(TokenizedCorpus);
              ReportStatistics;
              HardPause; }
              RunEmbed(TokenizedCorpus)
            end;
          end;
        end;
      end;
      {'5': begin
        write('Token file name: ');
        Readln(TokenFileName);
        FromSymbolTable := False;
        if not FileExists(TokenFileName) then begin
          Writeln('Token file not found: ', TokenFileName);
          Pause;
        end
        else begin
          LoadTokenList(TokenFileName);
          write('Output symbol table file name: ');
          if not FileExists(SymbolFileName) then begin
            Writeln('Symbol table file not found: ', SymbolFileName);
            Pause;
          end
          else begin
            Readln(SymbolFileName);
            LoadSymbolTable(SymbolFileName);
            RunEmbed(TokenizedCorpus);
            Pause;
          end;
        end;
      end;}
      '5': begin
        // Ask user for input file.
        Write('Enter input corpus filename: ');
        Readln(CorpusFileName);

        // Read bytes from file.
        if not FileExists(CorpusFileName) then begin
          SetLength(CorpusFileNames, 0);
          CorpusFileNames[0] := 'bela.txt';
          WorkingName := ChangeFileExt(CorpusFileName, '');
          CorpusFileName := 'bela.txt';
          writeln('File not found. Using bela.txt.');
        end
        else begin
          SetLength(CorpusFileNames, 0);
          CorpusFileNames[0] := CorpusFileName;
        end;

        RunGPT2Tokenize(CorpusFileName, TokenizedCorpus);
        writeln('First 200 token of tokenized corpus: ');
        for i := 0 to 199 do
          write(TokenizedCorpus[i], ' ');
        writeln;
        Pause;
        if nSymbols > 0 then
          RunEmbed(TokenizedCorpus)
        else
          writeln('Symbols not found in table.');
      end;
      '6': begin
        // Ask user for input file.
        Write('Enter input corpus filename: ');
        Readln(CorpusFileName);

        // Read bytes from file.
        if FileExists(CorpusFileName) then begin
          SetLength(CorpusFileNames, 0);
          WorkingName := ChangeFileExt(CorpusFileName, '');
          CorpusFileNames[0] := CorpusFileName;
          RunGPT2Tokenize(CorpusFileName, TokenizedCorpus);
          writeln('First 200 token of tokenized corpus: ');
          for i := 0 to 199 do
            write(TokenizedCorpus[i], ' ');
          writeln;
          Pause;
          if nSymbols > 0 then
            RunEmbed(TokenizedCorpus)
          else
            writeln('Symbols not found in table.');
        end
        else
          writeln('File not found: ', FileName, '.');
      end;
      '7': writeln('Not yet implemented');
      '9': begin
        // Ask user for input file.
        Write('Enter input filename: ');
        Readln(CorpusFileName);

        // Read bytes from file.
        if FileExists(CorpusFileName) then begin
          ReadFileBytes(CorpusFileName, Corpus);
          SetLength(CorpusFileNames, 1);
          CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
           + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

          FromSymbolTable := False;
          WorkingName := ChangeFileExt(CorpusFileName, '');
              writeln('in main workingname ', workingname, ' ', stamp); pause;
          RunSymbolize(Corpus);
          {if nSymbols > 0 then
            RunEmbed(TokenizedCorpus)
          else
            writeln('Symbols not found in table.');}
          DisplayByteSymbolTable(SymbolTable);
        end
        else
          writeln('File not found: ', FileName, '.');
      end;
      'X': Exit;
      'H': Help;
      'VTO': VerboseTokenize := True;
      'VV': VeryVerbose := True;
      'VTR': VerboseTransform := True;
      'NVTO': VerboseTokenize := False;
      'NVV': VeryVerbose := False;
      'NVTR': VerboseTransform := False;
      'DNP': DoNotPause := True;
      'DP': DoNotPause := False;
      'SF': SaveFiles := True;
      'NSF': SaveFiles := False;
      'M': begin
        write('Maximum merges: ');
        readln(MaxMerges);
      end;
      'PC': begin
        write('Maximum pair count: ');
        readln(MaxPairCount);
      end;
      'LR': begin
        write('Learning rate: ');
        readln(LearningRate);
      end
      else writeln('Invalid input');
    end;
  end;

end.

