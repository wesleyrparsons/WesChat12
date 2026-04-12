program WesChat;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}
{ Note: Edited 4/12/2026 4 pm}
{ Notes: TC comes from WesSymbolizeor or ChatGPTTokenize; let's make WesModel come from Embed }
uses
  CombineTables,
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
  Windows, Unit1;

var
  Corpus: TBVector;                    // Vector of byte.
  Ch: string;                          // For option menu.
  CorpusFileName, SymbolFileName,      // File names.
    TokenFileName, ListFile: string;
  CombinedSymbolTable: TSymbolTable;   // For combining two symbol tables.
  MinSymbols: Integer = 50;            // Minimum for loading.
  MinTokens: Integer = 50;             // Minimum for loading.
  MinCorpus: Integer = 50;             // Minimum for loading.

// Create and name directory and file for saving.
Procedure LogFile(const Eponym: string);
var
  SaveOut: Text;                            // Save Output mode.
begin
  WorkingDir := ChangeFileExt(Eponym, '') + FormatDateTime('yyyy-mm-dd_hhnnss', Now);
  WorkingName := WorkingDir;
  CreateDir(WorkingDir);                    // Create folder of files.
  ChDir(WorkingDir);                        // And go there.

  // Save current Output.
  SaveOut := Output;

  // Redirect Output.
  Assign(Output, WorkingName + '.log');     // Create log file in folder.
  ReWrite(Output);
  ReportInfo;                               // Write report of info in folder.

  // Restore Output to console.
  Close(Output);
  Output := SaveOut;                        // Go back to console.
  ChDir('..');                              // Go back to parent directory.
 end;

// Read a file of file names, and sends each to tokenizer.
procedure ProcessFileList(var ListFile: string; var Corpus: TBVector);
var
  F: TextFile;               // ListFile is the file of corpus file names.
  Line: string;              // Line is one corpus file name.
  FilesRead: TSVector;       // List of file names read.
  OneCorpus: TBVector;       // One corpus to concatenate.
  Count: Integer;
begin
  MultipleFileName := EmptyStr;        // This var contains info on input corpuses.
  Write('Enter name of file list: ');
  Readln(ListFile);
  if not FileExists(ListFile) then begin
    Writeln('List file not found: ', ListFile);
    Exit;
  end;

  AssignFile(F, ListFile);
  Reset(F);

  Count := 0;                          // Count the input corpuses.
  SetLength(FilesRead, 0);
  FromSymbolTable := False;            // Tells whether there's a symboltable.
  SetLength(Corpus, 0);                // Replace with length(ST)?

  while not EOF(F) do begin            // Loop thru the corpuses.
    ReadLn(F, Line);
    Line := Trim(Line);
    if Line = '' then Continue;         // Skip blank lines.
    if not FileExists(Line) then begin
      Writeln('  File not found: ', Line, '.');
      Continue;
    end;
    if (Count = 0) and SaveFiles then
      LogFile('Mult' + ListFile);

    ReadFileBytes(Line, OneCorpus);     // Read the file into OneCorpus.
    SetLength(CorpusFileNames, Count + 1);
    CorpusFileNames[Count] := Line;
    Writeln('  File processed: ', Line, '; corpus bytes read: ', Length(OneCorpus));
    if Length(OneCorpus) < MinCorpus then begin
      Writeln('Corpus too small. Aborting...');
      Continue;
    end;

    Corpus := Concat(Corpus, OneCorpus);     // Concat Corpus with OneCorpus.
    nCorpus := Length(Corpus);
    Writeln('Total bytes read: ', Length(Corpus));
    Inc(Count);
    SetLength(FilesRead, Count);
    FilesRead[Count - 1] := Line;
  end;

  CloseFile(F);

  Writeln('Combined corpus length = ', Length(Corpus));
  nCorpus := Length(Corpus);
  Pause;
end;

// Help file.
procedure Help;
begin
  Writeln('  VTO: VerboseTokenize := True');
  Writeln('  VV: VeryVerbose := True');
  Writeln('  VTR: VerboseTransform := True');
  Writeln('  NVTO: VerboseTokenize := False');
  Writeln('  NVV: VeryVerbose := False');
  Writeln('  NVTR: VerboseTransform := False');
  Writeln('  DNP: DoNotPause := True');
  Writeln('  DP: DoNotPause := False');
  Writeln('  SF: SaveFiles := True');
  Writeln('  NSF: SaveFiles := False');
  Writeln('  M: Maximum merges: ');
  Writeln('  PC: Maximum pair count: ');
  Writeln('  LR: Learning rate: ');
  Writeln;
end;

// Helper function for proceeding to Embed.
function QueryEmbed: Boolean;
begin
  Write('Do you wish to proceed to training? (y/n) ');
  Readln(Ch);
  if UpCase(Ch) = 'Y' then
    Result := True
  else
    Result := False;
end;

// Start of main program.
begin
  { Necessary because JSON will throw dupe errors otherwise }
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);

  { Below is not working on my Lazarus console }
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  Writeln('WesChat, Version 1.2, begun January 19, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.');
  Writeln;
  Writeln('Options:');
  Writeln('  1: Tokenize an input corpus from a file using WesChat''s byte-level byte-pair encoding, with');
  Writeln('     deterministic left-to-right longest-prefix matching and greedy longest-match decoding.');
  Writeln('  2: Tokenize an input set of corpuses listed one per line in a file, using WesChat''s tokenization routine,');
  Writeln('     to create a concatenated token list.');
  Writeln('  3: Tokenize Bela corpus using WesChat''s Bela symbol table.');
  Writeln('  4: Tokenize an input corpus, based on an input symbol table, using WesChat''s tokenization routine.');
  Writeln('  5: Tokenize an input corpus using ChatGPT''s symbol and merge tables and WesChat''s');
  Writeln('     tokenization routine.');
  Writeln('  6: Input a token list to be used in training.');
  Writeln('  7: Combine two symbol tables for use with WesChat''s tokenization routine.');
  Writeln('  8: Tokenize an input set of corpuses listed one per line in a file, using an input symbol table,');
  Writeln('     to create a concatenated token list.');
  Writeln('  9: Create symbol table from input corpus.');
  Writeln('  10: Input a weight model for funning forward.');
  Writeln('  11: Run a model forward.');
  Writeln('  H: Help.');
  Writeln('  X: Exit.');
  Writeln;
  Writeln('The symbol table and other information, including if desired the token list, will be written to disk.');
  Writeln('Ater tokenization, WesChat prompts for training the transformer, which consists');
  Writeln('of 4 to 8 blocks. The attention stage has 8 heads. There are a weight stage wih a bias');
  Writeln('and a weight stage without a bias. The activation function is softmax with temperature.');
  Writeln('Model dimensions are 160 or 256. The activation stage expands dimensionality fourfold.');
  Writeln('Precision is single. Sequence length is 128 or 256 bytes. Pre-layer normalization');
  Writeln('standardizes for means and standard deviations. Attention and residual dropouts are 0.1.');
  Writeln('The softmax function normalizes exponentially with a temperature of 1.0. The learning rate is 0.01.');
  Writeln('All output files will be contained in a folder or file named with the input file name,');
  Writeln('appended with a timestamp.');
  while True do begin
    Write('W>');
    Readln(Ch);
    Case UpperCase(Ch) of
      '1': begin
        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file .
        if not FileExists(CorpusFileName) then begin
          Writeln('  File not found: ', CorpusFileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read corpus bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
         + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Rum WesChat symbolizer.
        RunSymbolize(Corpus);
        RunWesTokenize(Corpus, TokenizedCorpus);

        // Check number of symbols.
        if nSymbols < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Embed.
        If QueryEmbed then
          RunEmbed(TokenizedCorpus);
      end;
      '2': begin
        // Process multiple corpuses.
        ProcessFileList(ListFile, Corpus);

        // Run WesChat symbolizer.
        RunSymbolize(Corpus);

        // Display symboltable.
        DisplayByteSymbolTable(SymbolTable);

        // Run WesChat tokenizer.
        RunWesTokenize(Corpus, TokenizedCorpus);
        If QueryEmbed then
          RunEmbed(TokenizedCorpus)
      end;
      '3': begin
        // Read corpus file.
        ReadFileBytes('bela.txt', Corpus);
        FromSymbolTable := True;
        nCorpus := Length(Corpus);

        // Read symbol table file.
        SymbolFileName := 'bela.sym';
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := SymbolFileName;

        // Read symboltable.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Write to log file.
        if SaveFiles then
          LogFile('bela.txt');

        // Run WesChat tokenizer.
        RunWesTokenize(Corpus, TokenizedCorpus);
        // Run Embed.
        If QueryEmbed then
            RunEmbed(TokenizedCorpus)
      end;
      '4': begin
        // Ask user for corpus file.
        Write('Input corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', CorpusFileName, '. Aborting...');
          Continue;
        end;
        if Length(Corpus) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Ask user for symbol file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;  // Do I need this var. Length(ST) = 0.

        // Check existence of symbol file.
        if not FileExists(SymbolFileName) then begin
          Writeln('File not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read the symbol table.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symbol table.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Read corpus bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName;

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run WesChat tokenizer.
        RunWesTokenize(Corpus, TokenizedCorpus);

        // Run Embed.
        If QueryEmbed then
            RunEmbed(TokenizedCorpus)
      end;
      '5': begin
        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check corpus file for existence and size.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', CorpusFileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        FromSymbolTable := True;
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName;

        // Write to log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run ChatGPT tokenizer.
        RunGPT2Tokenize(CorpusFileName, TokenizedCorpus);

        // Check tokenized corpus.
        Writeln('First 200 token of tokenized corpus: ');
        for i := 0 to 199 do
          Write(TokenizedCorpus[i], ' ');
        Writeln;
        Pause;

        // Check number of symbols, and Embed.
        if nSymbols > 0 then
          RunEmbed(TokenizedCorpus)
        else
          Writeln('Symbols not found in table.');
      end;
      '6': begin
        // Ask user for token file.
        Write('Enter token list file name: ');
        Readln(TokenFileName);

        // Check existence and size of token file.
        if not FileExists(TokenFileName) then begin
          Writeln('File not found: ', FileName, '.');
          Continue;
        end;

        // Read token file.
        IOHandler.LoadTokenList(TokenFileName, TokenizedCorpus);

        // Check size of token file.
        if Length(TokenizedCorpus) < MinTokens then begin
          Writeln('Token list too small. Aborting...');
          Continue;
        end;

        // Run Embed.
        If QueryEmbed then
          RunEmbed(TokenizedCorpus)
      end;
      '7': begin
        // Merge symbol tables.
        MergeSymbolTables(CombinedSymbolTable);

        // Ask user for output symbol table name.
        Write('Output symbol table name:');
        Readln(SymbolFileName);

        // Write to log file.
        if SaveFiles then
          LogFile(SymbolFileName);

        // Save combined symboltable.
        SaveSymbolTable(SymbolFileName, CombinedSymbolTable);
        Writeln('File ', SymbolFileName, ' successfully saved.');
        Writeln;
      end;
      '8': begin
        // Process multiple corpuses.
        ProcessFileList(ListFile, Corpus);

        // Ask user for symbol table file.
        Write('Input symbol table file name: ');
        Readln(SymbolFileName);
        FromSymbolTable := True;

        // Check for existence of symboltable.
        if not FileExists(SymbolFileName) then begin
          Writeln('Symbol table file not found: ', SymbolFileName, '. Aborting...');
          Continue;
        end;

        // Read symboltable.
        LoadSymbolTable(SymbolFileName, SymbolTable);

        // Check size of symboltable.
        if Length(SymbolTable) < MinSymbols then begin
          Writeln('Too few symbols found. Aborting...');
          Continue;
        end;

        // Display symboltable.
        DisplayByteSymbolTable(SymbolTable);

        // Run WesChat tokenizer.
        RunWesTokenize(Corpus, TokenizedCorpus);
        If QueryEmbed then
          RunEmbed(TokenizedCorpus)
      end;
      '9': begin
        // Ask user for corpus file.
        Write('Enter corpus file name: ');
        Readln(CorpusFileName);

        // Check existence and size of corpus file.
        if not FileExists(CorpusFileName) then begin
          Writeln('File not found: ', FileName, '.');
          Continue;
        end;
        if FileSize(CorpusFileName) < MinCorpus then begin
          Writeln('Corpus too small. Aborting...');
          Continue;
        end;

        // Read bytes from file.
        ReadFileBytes(CorpusFileName, Corpus);
        nCorpus := Length(Corpus);
        SetLength(CorpusFileNames, 1);
        CorpusFileNames[0] := CorpusFileName + '   ' + IntToStr(FileSize(CorpusFileName))
          + ' bytes   ' + DateTimeToStr(FileDateToDateTime(FileAge(CorpusFileName)));

        // Write to Log file.
        if SaveFiles then
          LogFile(CorpusFileName);

        // Run WesChat symbolizer.
        RunSymbolize(Corpus);

        // Check number of symbols, and Embed.
        if nSymbols > MinSymbols then
          RunEmbed(TokenizedCorpus)
        else
          Writeln('Symbols not found in table.');

        // Display symbol table.
        DisplayByteSymbolTable(SymbolTable);
      end;
      '10', '11': Writeln('Not yet available.');
      'X':     Exit;
      'H':     Help;
      'VTO':   VerboseTokenize := True;
      'VV':    VeryVerbose := True;
      'VTR':   VerboseTransform := True;
      'NVTO':  VerboseTokenize := False;
      'NVV':   VeryVerbose := False;
      'NVTR':  VerboseTransform := False;
      'DNP':   DoNotPause := True;
      'DP':    DoNotPause := False;
      'SF':    SaveFiles := True;
      'NSF':   SaveFiles := False;
      'M': begin
        Write('Maximum merges: ');
        Readln(MaxMerges);
      end;
      'PC': begin
        Write('Maximum pair count: ');
        Readln(MaxPairCount);
      end;
      'LR': begin
        Write('Learning rate: ');
        Readln(LearningRate);
      end
      else Writeln('Invalid input');
    end;
  end;
end.

