unit Matrix;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}
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
  cublasDLL = 'cublas64_13.dll';
  cudartDLL = 'cudart64_13.dll';
  copenblasDLL = 'libopenblas.dll';
  cudaMemcpyHostToHost   = 0;
  cudaMemcpyHostToDevice = 1;
  cudaMemcpyDeviceToHost = 2;
  cudaMemcpyDeviceToDevice = 3;

// cublas functions.
function cublasCreate_v2(out handle: TcublasHandle): Integer; cdecl; external cublasDLL;
function cublasDestroy_v2(handle: TcublasHandle): Integer; cdecl; external cublasDLL;
function cudaMalloc(devPtr: PPointer; size: NativeUInt): Integer; cdecl; external cudartDLL;
function cudaMemcpy(dst: Pointer; src: Pointer; count: NativeUInt; kind: LongInt): LongInt; cdecl; external cudartDLL;
function cudaMemset(devPtr: Pointer; value: Integer; count: NativeUInt): Integer; cdecl; external cudartDLL;
function cudaFree(devPtr: Pointer): Integer; cdecl; external cudartDLL;
function cudaDeviceReset: Integer; cdecl; external cudartDLL;
function cudaGetLastError: Integer; cdecl; external cudartDLL;
function cudaDeviceSynchronize: Integer; cdecl; external cudartDLL;

// Multiply and add procedures.
// procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullNN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
// procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullTN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
// procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullNT(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
// procedure MatMulFullAccNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
// procedure MatMulFullAccNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
// procedure MatMulFullAccTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
procedure CuMatMulFullAccTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
// procedure MatMulFullScaledNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
procedure CuMatMulFullScaledNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
// procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
// procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
// procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
// procedure AddScaled(const N: Integer; const Alpha: Single; const X: PSingle; Y: PSingle);
procedure CuAddScaled(handle: TcublasHandle; N: Integer; Alpha: Single; const X: PSingle; Y: PSingle);
// procedure Scale(const N: Integer; const Alpha: Single; X: PSingle);
procedure CuScale(handle: TcublasHandle; N: Integer; Alpha: Single; X: PSingle);
// procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
procedure CuMatAdd(handle: TcublasHandle; const A, B: PSingle; C: PSingle; Rows, Cols: Integer);

// Split and accumulate procedures.
// procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
procedure CuGradSplit(handle: TcublasHandle; const Upstream: PSingle; Left, Right: PSingle; Rows, Cols: Integer);
// procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
procedure CuAccumulateGrad(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; Rows, Cols: Integer);
// procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
// procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
procedure CuMatMulAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);

// ReLU procedure.
// procedure ReLUMaskForward(const A: THiddenMatrix; var B: THiddenMatrix);
procedure LaunchReLUForward(A: PSingle; B: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';
procedure LaunchReLUBackward(Hidden1: PSingle; GradOut: PSingle; GradIn: PSingle; Rows: Integer; Cols: Integer);
  cdecl; external 'WesChatKernel12.dll';

// Copy matrix procedure.
// procedure CopyXTensor(const A: TSeqTensor; var B: TSeqTensor);
procedure CuCopyXTensor(handle: TcublasHandle; const A_Value, A_Grad: PSingle; B_Value, B_Grad: PSingle; SeqLen, ModelDim: Integer);

// sgemm.
// cblas.
procedure cblas_sgemm(Layout: LongInt;
  TransA: LongInt; TransB: LongInt;
  M: TMKLInt; N: TMKLInt; K: TMKLInt;
  Alpha: Single;
  const A: PSingle; LDA: TMKLInt;
  const B: PSingle; LDB: TMKLInt;
  Beta: Single;
  C: PSingle;  LDC: TMKLInt); cdecl; external copenblasDLL;

// cublas.
function cublasSgemm_v2(handle: TcublasHandle;
  transa, transb: Integer;
  m, n, k: Integer;
  const alpha: PSingle;
  A: PSingle; lda: Integer;
  B: PSingle; ldb: Integer;
  const beta: PSingle;
  C: PSingle; ldc: Integer): Integer; cdecl; external cublasDLL;

// saxpy.
// cblas.
procedure cblas_saxpy(N: LongInt;
  alpha: Single;
  X: PSingle; incX: LongInt;
  Y: PSingle; incY: LongInt); cdecl; external copenblasDLL;

// cublas.
function cublasSaxpy_v2(handle: TcublasHandle;
  n: LongInt;
  const alpha: PSingle;
  const x: PSingle; incx: LongInt;
  y: PSingle; incy: LongInt): LongInt; cdecl; external cublasDLL;

// scopy.
// cblas.
procedure cblas_scopy(N: LongInt;
  const X: PSingle; incX: LongInt;
  Y: PSingle; incY: LongInt); cdecl; external copenblasDLL;

// cublas.
function cublasScopy_v2(handle: TcublasHandle;
  n: LongInt;
  const x: PSingle; incx: LongInt;
  y: PSingle; incy: LongInt): LongInt; cdecl; external cublasDLL;

// sscal.
// cblas.
procedure cblas_sscal(N: LongInt;
  alpha: Single; X: PSingle;
  incX: LongInt); cdecl; external copenblasDLL;

// cublas.
function cublasSscal_v2(handle: TcublasHandle;
  n: LongInt;
  const alpha: PSingle;
  x: PSingle;
  incx: LongInt): LongInt; cdecl; external cublasDLL;

// sdot.
// cblas.
function cblas_sdot(N: LongInt;
  const X: PSingle; incX: LongInt;
  const Y: PSingle; incY: LongInt): Single; cdecl; external copenblasDLL;

// snrm2.
// cblas.
function cblas_snrm2(N: LongInt;
  const X: PSingle; incX: LongInt): Single; cdecl; external copenblasDLL;

implementation

// Future four-way.
{procedure Axpy(N: Integer; Alpha: Float; X, Y: PFloat);
begin
{$ifdef USE_CUDA}
  {$ifdef USE_DOUBLE}
    cublasDaxpy_v2(handle, N, @Alpha, X, 1, Y, 1);
  {$else}
    cublasSaxpy_v2(handle, N, @Alpha, X, 1, Y, 1);
  {$endif}
{$else}
  {$ifdef USE_DOUBLE}
    cblas_daxpy(N, Alpha, X, 1, Y, 1);
  {$else}
    cblas_saxpy(N, Alpha, X, 1, Y, 1);
  {$endif}
{$endif}
end;}

// Split Gradient into 2 streams, for backprop.
// cblas.
procedure GradSplit(const Upstream: TSeqMatrix; var Left, Right: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // Left += Upstream.
  cblas_saxpy(n, 1.0, @Upstream[0,0], 1, @Left[0,0], 1);

  // Right += Upstream.
  cblas_saxpy(n, 1.0, @Upstream[0,0], 1, @Right[0,0], 1);
end;

procedure CuGradSplit(handle: TcublasHandle; const Upstream: PSingle; Left, Right: PSingle; Rows, Cols: Integer);
// cublas.
var
  n: Integer;
  alpha: Single;
begin
  n := Rows * Cols;
  alpha := 1.0;

  // Left += Upstream
  cublasSaxpy_v2(handle, n, @alpha, Upstream, 1, Left, 1);

  // Right += Upstream
  cublasSaxpy_v2(handle, n, @alpha, Upstream, 1, Right, 1);
end;

// Accummulate Gradient.
// cblas.
procedure AccumulateGrad(const Src: TSeqMatrix; var Dst: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;
  cblas_saxpy(n, 1.0, @Src[0,0], 1, @Dst[0,0], 1);
end;

// cublas.
procedure CuAccumulateGrad(handle: TcublasHandle; const Src: PSingle; Dst: PSingle; Rows, Cols: Integer);
var
  n: Integer;
  alpha: Single;
begin
  n := Rows * Cols;
  alpha := 1.0;
  cublasSaxpy_v2(handle, n, @alpha, Src, 1, Dst, 1);
end;

// Full matrix multiplication (lda, ldb, ldc), A no transpose, B no transpose, overwrite, row-major.
procedure MatMulFullNN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans, M, N, K, 1.0,
    A, lda, B, ldb, 0.0, C, ldc);
end;

// Cublas.
procedure CuMatMulFullNN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 0.0;

  cublasSgemm_v2(Handle, 0, 0, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;

// Full matrix multiplication (lda, ldb, ldc), A no transpose, B transpose, overwrite, row-major.
procedure MatMulFullNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1.0,
    A, lda, B, ldb, 0.0, C, ldc);
end;

// cublas.
procedure CuMatMulFullNT(
  Handle: TcublasHandle;
  const A, B: PSingle;
  C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 0.0;

  // Row-major C = A * B^T
  // Column-major equivalent: C^T = B * A^T
  cublasSgemm_v2(Handle, 1, 0, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;
// Full matrix multiplication (lda, ldb, ldc), A transpose, B no transpose, overWrite, row-major.
procedure MatMulFullTN(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans, M, N, K, 1.0,
    A, lda, B, ldb, 0.0, C, ldc);
end;

// cublas.
procedure CuMatMulFullTN(Handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 0.0;
  cublasSgemm_v2(Handle, 1, 0, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;

// Full matrix multiply, A no transpose, B no transpose, accumulate.
// C := C + A * B
procedure MatMulFullAccNN(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans, M, N, K, 1.0,
    A, lda, B, ldb, 1.0, C, ldc);
end;

procedure CuMatMulFullAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer;
  lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 1.0;   // accumulate into C.

  cublasSgemm_v2(handle, 0, 0, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;

// Full matrix multiply, A no transpose, B transpose, accumulate.
// C := C + A * B^T
procedure MatMulFullAccNT(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1.0,
    A, lda, B, ldb, 1.0, C, ldc);
end;

procedure CuMatMulFullAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  cublasSgemm_v2(handle, 0, 1, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;

// Full matrix multiply, A transpose, B no transpose, accumulate.
// C := C + A^T * B
procedure MatMulFullAccTN(const A, B: PSingle; C: PSingle;
  M, N, K, lda, ldb, ldc: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans, M, N, K, 1.0,
    A, lda, B, ldb, 1.0, C, ldc);
end;

procedure CuMatMulFullAccTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer; lda, ldb, ldc: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  cublasSgemm_v2(handle, 1, 0, N, M, K, @alpha,
    B, ldb, A, lda, @beta, C, ldc);
end;

// Matrix multiplication, A no transpose, B transpose, scaled overwrite, row-major.
procedure MatMulFullScaledNT(const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, Alpha,
    A, lda, B, ldb, Beta, C, ldc);
end;

// cublas.
procedure CuMatMulFullScaledNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K, lda, ldb, ldc: Integer; Alpha, Beta: Single);
begin
  cublasSgemm_v2(handle, 0, 1, N, M, K, @Alpha,
    B, ldb, A, lda, @Beta, C, ldc);
end;

// Matrix multiplication, A no transpose, B no transpose, overwrite, row-major.
procedure MatMulNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans, M, N, K, 1.0,
    A, K, B, N, 0.0, C, N);
end;

procedure CuMatMulNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 0.0;   // Overwrite C.

  cublasSgemm_v2(handle, 0, 0, N, M, K, @alpha,
    B, N, A, K,@beta, C, N);
end;

// Matrix multiplication, A no transpose, B transpose, overwrite, row-major.
procedure MatMulNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1.0,
    A, K, B, K, 0.0, C, N);
end;

procedure CuMatMulNT(handle: TcublasHandle;
  const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  CuMatMulFullNT(handle, A, B, C, M, N, K, K, K, N);
end;

// Matrix multiplication, A transpose, B no transpose, overwrite, row-major.
procedure MatMulTN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, Trans, NoTrans, M, N, K, 1.0,
    A, M, B, N, 0.0, C, N);
end;

procedure CuMatMulTN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 0.0;   // Overwrite C.

  cublasSgemm_v2(handle, 1, 0, N, M, K, @alpha,
    B, N, A, M, @beta, C, N);
end;

// Matrix multiplication, A no transpose, B no transpose, accumulate, row-major.
// cblas.
procedure MatMulAccNN(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, NoTrans, M, N, K, 1.0,
    A, K, B, N, 1.0, C, N);
end;

// cublas.
procedure CuMatMulAccNN(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  cublasSgemm_v2(handle, 0, 0, N, M, K, @alpha,
    B, N, A, K, @beta, C, N);
end;

// Matrix multiplication, A no transpose, B transpose, accumulate, row-major.
// cblas.
procedure MatMulAccNT(const A, B: PSingle; C: PSingle; M, N, K: Integer);
begin
  cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1.0,
    A, K, B, K, 1.0, C, N);
end;

// cublas.
procedure CuMatMulAccNT(handle: TcublasHandle; const A, B: PSingle; C: PSingle; M, N, K: Integer);
var
  alpha, beta: Single;
begin
  alpha := 1.0;
  beta  := 1.0;   // Accumulate into C.

  cublasSgemm_v2(handle, 0, 1, N, M, K, @alpha,
    B, K, A, K, @beta, C, N);
end;

// Add scaled vector.
procedure AddScaled(const N: Integer; const Alpha: Single; const X: PSingle; Y: PSingle);
begin
  cblas_saxpy(N, Alpha, X, 1, Y, 1);
end;

procedure CuAddScaled(handle: TcublasHandle; N: Integer; Alpha: Single; const X: PSingle; Y: PSingle);
begin
  // Performs: Y[i] := Alpha * X[i] + Y[i].
  cublasSaxpy_v2(handle, N, @Alpha, X, 1, Y, 1);
end;

// Scale vector.
procedure Scale(const N: Integer; const Alpha: Single; X: PSingle);
begin
  cblas_sscal(N, Alpha, X, 1);
end;

procedure CuScale(handle: TcublasHandle; N: Integer; Alpha: Single; X: PSingle);
begin
  // Performs: X[i] := Alpha * X[i].
  cublasSscal_v2(handle, N, @Alpha, X, 1);
end;

// Matrix addition, overwrite.
procedure MatAdd(const A, B: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // C := A.
  cblas_scopy(n, @A[0,0], 1, @C[0,0], 1);
  // C += B.
  cblas_saxpy(n, 1.0, @B[0,0], 1, @C[0,0], 1);
end;

procedure CuMatAdd(handle: TcublasHandle;  const A, B: PSingle; C: PSingle; Rows, Cols: Integer);
var
  n: Integer;
  alpha: Single;
begin
  n := Rows * Cols;

  // C := A
  cublasScopy_v2(handle, n, A, 1, C, 1);
  // C += B
  alpha := 1.0;

  cublasSaxpy_v2(handle, n, @alpha, B, 1, C, 1);
end;

// Matrix addition, accumulate.
procedure MatAccumulate(const A: TSeqMatrix; var C: TSeqMatrix; Rows, Cols: Integer);
var
  n: Integer;
begin
  n := Rows * Cols;

  // C += A.
  cblas_saxpy(n, 1.0, @A[0,0], 1, @C[0,0], 1);
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
procedure CopyXTensor(const A: TSeqTensor; var B: TSeqTensor);
begin
  cblas_scopy(SeqLen * ModelDim, @A.Value[0,0], 1, @B.Value[0,0], 1);
  cblas_scopy(SeqLen * ModelDim, @A.Grad[0,0], 1, @B.Grad[0,0], 1);
end;

procedure CuCopyXTensor(handle: TcublasHandle; const A_Value, A_Grad: PSingle; B_Value, B_Grad: PSingle; SeqLen, ModelDim: Integer);
var
  n: Integer;
begin
  n := SeqLen * ModelDim;

  // B.Value := A.Value.
  cublasScopy_v2(handle, n, A_Value, 1, B_Value, 1);

  // B.Grad := A.Grad.
  cublasScopy_v2(handle, n, A_Grad, 1, B_Grad, 1);
end;

end.

