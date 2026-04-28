unit Matrix;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Global,
  Math;

type
  TMKLInt = LongInt;    // MKL_INT is 32-bit int in CBLAS interface.

const
  RowMajor = 101;       // Row Major.
  NoTrans  = 111;       // No transposition.
  Trans    = 112;       // Transposition.

// Multiply and add procedures.
procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullAccNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullAccNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullAccTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure MatMulFullScaledNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure AddScaled(const N: Integer; const Alpha: Single; const X: PSingle; Y: PSingle);
procedure Scale(const N: Integer; const Alpha: Single; X: PSingle);
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);

// Split and accumulate procedures.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);

// ReLU procedure.
procedure ReLUMaskForward(const A: THiddenMatrix; var B: THiddenMatrix);

// Copy matrix procedure.
procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector;
  const Rows, Cols: Integer);
procedure FastCopyXMatrix(const A: TSeqMatrix; var B: TSeqMatrix);

// cblas sgemm.
procedure cblas_sgemm(Layout: LongInt;
  TransA: LongInt; TransB: LongInt;
  M: TMKLInt; N: TMKLInt; K: TMKLInt;
  Alpha: Single;
  const A: PSingle; LDA: TMKLInt;
  const B: PSingle; LDB: TMKLInt;
  Beta: Single;
  C: PSingle;  LDC: TMKLInt); cdecl; external 'libopenblas.dll';

// cblas saxpy.
procedure cblas_saxpy(N: LongInt;
  alpha: Single;
  X: PSingle; incX: LongInt;
  Y: PSingle; incY: LongInt); cdecl; external 'libopenblas.dll';

// cblas scopy.
procedure cblas_scopy(N: LongInt;
  const X: PSingle; incX: LongInt;
  Y: PSingle; incY: LongInt); cdecl; external 'libopenblas.dll';

// cblas sscal.
procedure cblas_sscal(N: LongInt;
  alpha: Single; X: PSingle;
  incX: LongInt); cdecl; external 'libopenblas.dll';

// cblas sdot.
function cblas_sdot(N: LongInt;
  const X: PSingle; incX: LongInt;
  const Y: PSingle; incY: LongInt): Single; cdecl; external 'libopenblas.dll';

// cblas snrm2.
function cblas_snrm2(N: LongInt;
  const X: PSingle; incX: LongInt): Single; cdecl; external 'libopenblas.dll';

implementation

// Split Gradient into 2 streams, for backprop.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // Left += Upstream.
  cblas_saxpy(n,
    1.0,
    @Upstream[0,0], 1,
    @Left[0,0], 1);

  // Right += Upstream.
  cblas_saxpy(n,
    1.0,
    @Upstream[0,0], 1,
    @Right[0,0], 1);
end;

// Accummulate Gradient.
procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;
  cblas_saxpy(n,
    1.0,
    @Src[0,0], 1,
    @Dst[0,0], 1);
end;

// Full matrix multiplication (lda, ldb, ldc), A no transpose, B no transpose, overwrite, row-major.
procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Full matrix multiplication (lda, ldb, ldc), A no transpose, B transpose, overwrite, row-major.
procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Full matrix multiplication (lda, ldb, ldc), A transpose, B no transpose, overWrite, row-major.
procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    0.0,
    C, ldc);
end;

// Full matrix multiply, A no transpose, B no transpose, accumulate.
// C := C + A * B
procedure MatMulFullAccNN(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    1.0,     // Accumulate.
    C, ldc);
end;

// Full matrix multiply, A no transpose, B transpose, accumulate.
// C := C + A^T * B
procedure MatMulFullAccNT(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    1.0,     // Accumulate.
    C, ldc);
end;

// Full matrix multiply, A transpose, B no transpose, accumulate.
// C := C + A * B^T
procedure MatMulFullAccTN(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans,
    M, N, K,
    1.0,
    A, lda,
    B, ldb,
    1.0,     // Accumulate.
    C, ldc);
end;

// Matrix multiplication, A no transpose, B transpose, scaled overwrite, row-major.
procedure MatMulFullScaledNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans,
    M, N, K,
    Alpha,
    A, lda,
    B, ldb,
    Beta,
    C, ldc);
end;

// Matrix multiplication, A no transpose, B no transpose, overwrite, row-major.
procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans,
    M, N, K,
    1.0,
    A, K,
    B, N,
    0.0,
    C, N);
end;

// Matrix multiplication, A no transpose, B transpose, overwrite, row-major.
procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans,
    M, N, K,
    1.0,
    A, K,
    B, K,
    0.0,
    C, N);
end;

// Matrix multiplication, A transpose, B no transpose, overwrite, row-major.
procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans,
    M, N, K,
    1.0,
    A, M,
    B, N,
    0.0,
    C, N);
end;

// Matrix multiplication, A no transpose, B no transpose, accumulate, row-major.
procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans,
    M, N, K,
    1.0,
    A, K,
    B, N,
    1.0,
    C, N);
end;

// Matrix multiplication, A no transpose, B transpose, accumulate, row-major.
procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans,
    M, N, K,
    1.0,
    A, K,
    B, K,
    1.0,
    C, N);
end;

// Add scaled vector.
procedure AddScaled(const N: Integer; const Alpha: Single; const X: PSingle; Y: PSingle);
begin
  cblas_saxpy(N,
    Alpha,
    X, 1,
    Y, 1
  );
end;

// Scale vector.
procedure Scale(const N: Integer; const Alpha: Single; X: PSingle);
begin
  cblas_sscal(
    N,
    Alpha,
    X,1
  );
end;

// Matrix addition, overwrite.
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // C := A.
  cblas_scopy(n,
    @A[0,0], 1,
    @C[0,0], 1);

  // C += B.
  cblas_saxpy(n,
    1.0,
    @B[0,0], 1,
    @C[0,0], 1);
end;

// Matrix addition, accumulate.
procedure MatAccumulate(const A: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // C += A.
  cblas_saxpy(n,
    1.0,
    @A[0,0], 1,
    @C[0,0], 1);
end;

// Apply ReLU to each item in a matrix.
procedure ReLUMaskForward(const A: THiddenMatrix; var B: THiddenMatrix);
var
  i, j: Integer;
begin
  for i:= 0 to High(A) do
    for j := 0 to High(A[0]) do
      B[i, j] := Max(0.0, A[i, j]);
end;

// Copy an X matrix. Not used.
procedure CopyXMatrix(const A: array of TSeqVector; var B: array of TSeqVector;
  const Rows, Cols: Integer);
var
  i: Integer;
begin
  if Rows <= 0 then Exit;
  if Cols <= 0 then Exit;

  for i := 0 to Rows - 1 do
    cblas_scopy(Cols, @A[i, 0], 1, @B[i, 0], 1);
end;

// Copy an X matrix, faster alternative.
procedure FastCopyXMatrix(const A: TSeqMatrix; var B: TSeqMatrix);
begin
  cblas_scopy(SeqLen * ModelDim, @A[0,0], 1, @B[0,0], 1);
end;

end.

