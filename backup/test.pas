unit Test;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Matrix;

type
  SASHLVector = array[0..SeqLen - 1] of Single;
  SASHDDArray = array[0..ModelDim - 1, 0..ModelDim - 1] of Single;
  SASHLDArray = array[0..SeqLen - 1, 0..ModelDim - 1] of Single;
  SASHDLArray = array[0..ModelDim - 1, 0..SeqLen - 1] of Single;
  SASHLLArray = array[0..SeqLen - 1, 0..SeqLen - 1] of Single;

procedure RunTestSGEMM;
{procedure BlockedMatMulLLLD(const A: SASHLLArray; B: SASHLDArray; out C: SASHLDArray);
procedure BlockedMatMulLDDL(const A: SASHLDArray; B: SASHDLArray; out C: SASHLLArray);
procedure BlockedMatMulLDDD(const A: SASHLDArray; B: SASHDDArray; out C: SASHLDArray);
function TransposeLD(const A: SASHLDArray): SASHDLArray;                              }

implementation

type
  TTestVector = array[0..99] of Single;

var
  TestVector: TTestVector;

procedure InitTestVector(var N: TTestVector);
var
  i: Integer;
begin
for i := 0 to 99 do
  N[i] := 0.0;
end;

procedure RunTestSGEMM;

var
  X: array[0..1, 0..2] of Single;   // 2×3
  Wq: array[0..2] of Single;  // 3×1
  OutVec: array[0..1] of Single; // 2×1

begin

  // Fill X row-major
  X[0, 0] := 1; X[0, 1] := 2; X[0, 2] := 3;
  X[1, 0] := 2; X[1, 1] := 3; X[1, 2] := 4;
  //X[0] := 1; X[1] := 2; X[2] := 3; X[3] := 2; X[4] := 3; X[5] := 4;

  // Fill Wq
  Wq[0] := 3.0; Wq[1] := 2.0; Wq[2] := 1.0;

  //OutVec[0] := 0.0;  OutVec[1] := 0.0;

  // SGEMM call, OpenBLAS 3.30
  cblas_sgemm(
    101,               // Layout
    111,               // A not transposed
    111,               // B not transposed
    2,                 // M = rows of X
    1,                 // N = columns of Wq
    3,                 // K = shared dimension
    1.0,               // alpha
    @X[0],             // A
    3,                 // LDA = K for row-major
    @Wq[0],            // B
    1,                 // LDB = N for row-major
    0.0,               // beta
    @OutVec[0],        // C
    1
  );

  writeln('Test OpenBLAS');
  writeln('X123 ', X[0, 0], ' ', X[0, 1], ' ', X[0, 2]);
  writeln('OutVec ', Outvec[0]:0:0, ' ', Outvec[1]:0:0);
  readln;

{  // SGEMM call, Intel MKL
  cblas_sgemm(
    101,               // Layout
    111,               // A not transposed
    111,               // B not transposed
    2,                 // M = rows of X
    1,                 // N = columns of Wq
    3,                 // K = shared dimension
    1.0,               // alpha
    @X[0],             // A
    3,                 // LDA = K for row-major
    @Wq[0],            // B
    1,                 // LDB = N for row-major
    0.0,               // beta
    @OutVec[0],        // C
    1
  );}

{  // SGEMM call based on AMD AOCL -- not working 1/20/2026
  aocl_gemm_bf16bf16f32obf16(
    'R',     // Layout
    'N',      // A not transposed
    'N',      // B not transposed
    2,                 // M = rows of X
    1,                 // N = columns of Wq
    3,                 // K = shared dimension
    1.0,               // alpha
    @X[0],             // A
    3,                 // LDA = K for row-major
    'R',               //mem_format_a,
    @Wq[0],            // B
    1,                 // LDB = N for row-major  item 12 illegal value????
    'R',
    0.0,               // beta
    @OutVec[0],        // C
    1,                          //????
    nil                // LDC = N
  ); }

  writeln('Test MKL BLAS');
  writeln('X123 ', X[0, 0], ' ', X[0, 1], ' ', X[0, 2]);
  writeln('OutVec ', Outvec[0]:0:0, ' ', Outvec[1]:0:0);
  readln;

end;

{type
  md_t = LongInt;
  dlp_metadata_t = Pointer;  // you can pass nil

AMD AOCL routine, as of 1/20/2026, Not working
procedure aocl_gemm_bf16bf16f32obf16(
  order: AnsiChar;
  transa: AnsiChar;
  transb: AnsiChar;
  m: md_t;
  n: md_t;
  k: md_t;
  alpha: Single;
   a: Pbfloat16;
  lda: md_t;
  mem_format_a: AnsiChar;
  b: Pbfloat16;
  ldb: md_t;
  mem_format_b: AnsiChar;
  beta: Single;
  c: Pbfloat16;
  ldc: md_t;
  metadata: dlp_metadata_t{a: PSingle;
  lda: md_t;
  mem_format_a: AnsiChar;
  b: PSingle;
  ldb: md_t;
  mem_format_b: AnsiChar;
  beta: Single;
  c: PSingle;
  ldc: md_t;
  metadata: dlp_metadata_t }
); cdecl; external 'AOCL-LibBlis-Win-dll.dll' name 'aocl_gemm_bf16bf16f32obf16';}


end.

