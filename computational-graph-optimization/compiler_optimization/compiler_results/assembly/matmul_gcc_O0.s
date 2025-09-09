	.file	"tmpaxpusegz.c"
# GNU C17 (Ubuntu 13.3.0-6ubuntu2~24.04) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=generic -march=x86-64 -O0 -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
	.globl	matrix_multiply_asm
	.type	matrix_multiply_asm, @function
matrix_multiply_asm:
.LFB0:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)	# A, A
	movq	%rsi, -32(%rbp)	# B, B
	movq	%rdx, -40(%rbp)	# C, C
	movl	%ecx, -44(%rbp)	# n, n
# /tmp/tmpaxpusegz.c:4:     for (int i = 0; i < n; i++) {
	movl	$0, -12(%rbp)	#, i
# /tmp/tmpaxpusegz.c:4:     for (int i = 0; i < n; i++) {
	jmp	.L2	#
.L7:
# /tmp/tmpaxpusegz.c:5:         for (int j = 0; j < n; j++) {
	movl	$0, -8(%rbp)	#, j
# /tmp/tmpaxpusegz.c:5:         for (int j = 0; j < n; j++) {
	jmp	.L3	#
.L6:
# /tmp/tmpaxpusegz.c:6:             C[i * n + j] = 0.0;
	movl	-12(%rbp), %eax	# i, tmp114
	imull	-44(%rbp), %eax	# n, tmp114
	movl	%eax, %edx	# tmp114, _1
# /tmp/tmpaxpusegz.c:6:             C[i * n + j] = 0.0;
	movl	-8(%rbp), %eax	# j, tmp115
	addl	%edx, %eax	# _1, _2
	cltq
# /tmp/tmpaxpusegz.c:6:             C[i * n + j] = 0.0;
	leaq	0(,%rax,8), %rdx	#, _4
	movq	-40(%rbp), %rax	# C, tmp116
	addq	%rdx, %rax	# _4, _5
# /tmp/tmpaxpusegz.c:6:             C[i * n + j] = 0.0;
	pxor	%xmm0, %xmm0	# tmp117
	movsd	%xmm0, (%rax)	# tmp117, *_5
# /tmp/tmpaxpusegz.c:7:             for (int k = 0; k < n; k++) {
	movl	$0, -4(%rbp)	#, k
# /tmp/tmpaxpusegz.c:7:             for (int k = 0; k < n; k++) {
	jmp	.L4	#
.L5:
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-12(%rbp), %eax	# i, tmp118
	imull	-44(%rbp), %eax	# n, tmp118
	movl	%eax, %edx	# tmp118, _6
	movl	-8(%rbp), %eax	# j, tmp119
	addl	%edx, %eax	# _6, _7
	cltq
	leaq	0(,%rax,8), %rdx	#, _9
	movq	-40(%rbp), %rax	# C, tmp120
	addq	%rdx, %rax	# _9, _10
	movsd	(%rax), %xmm1	# *_10, _11
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-12(%rbp), %eax	# i, tmp121
	imull	-44(%rbp), %eax	# n, tmp121
	movl	%eax, %edx	# tmp121, _12
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-4(%rbp), %eax	# k, tmp122
	addl	%edx, %eax	# _12, _13
	cltq
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	leaq	0(,%rax,8), %rdx	#, _15
	movq	-24(%rbp), %rax	# A, tmp123
	addq	%rdx, %rax	# _15, _16
	movsd	(%rax), %xmm2	# *_16, _17
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-4(%rbp), %eax	# k, tmp124
	imull	-44(%rbp), %eax	# n, tmp124
	movl	%eax, %edx	# tmp124, _18
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-8(%rbp), %eax	# j, tmp125
	addl	%edx, %eax	# _18, _19
	cltq
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	leaq	0(,%rax,8), %rdx	#, _21
	movq	-32(%rbp), %rax	# B, tmp126
	addq	%rdx, %rax	# _21, _22
	movsd	(%rax), %xmm0	# *_22, _23
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	mulsd	%xmm2, %xmm0	# _17, _24
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	movl	-12(%rbp), %eax	# i, tmp127
	imull	-44(%rbp), %eax	# n, tmp127
	movl	%eax, %edx	# tmp127, _25
	movl	-8(%rbp), %eax	# j, tmp128
	addl	%edx, %eax	# _25, _26
	cltq
	leaq	0(,%rax,8), %rdx	#, _28
	movq	-40(%rbp), %rax	# C, tmp129
	addq	%rdx, %rax	# _28, _29
# /tmp/tmpaxpusegz.c:8:                 C[i * n + j] += A[i * n + k] * B[k * n + j];
	addsd	%xmm1, %xmm0	# _11, _30
	movsd	%xmm0, (%rax)	# _30, *_29
# /tmp/tmpaxpusegz.c:7:             for (int k = 0; k < n; k++) {
	addl	$1, -4(%rbp)	#, k
.L4:
# /tmp/tmpaxpusegz.c:7:             for (int k = 0; k < n; k++) {
	movl	-4(%rbp), %eax	# k, tmp130
	cmpl	-44(%rbp), %eax	# n, tmp130
	jl	.L5	#,
# /tmp/tmpaxpusegz.c:5:         for (int j = 0; j < n; j++) {
	addl	$1, -8(%rbp)	#, j
.L3:
# /tmp/tmpaxpusegz.c:5:         for (int j = 0; j < n; j++) {
	movl	-8(%rbp), %eax	# j, tmp131
	cmpl	-44(%rbp), %eax	# n, tmp131
	jl	.L6	#,
# /tmp/tmpaxpusegz.c:4:     for (int i = 0; i < n; i++) {
	addl	$1, -12(%rbp)	#, i
.L2:
# /tmp/tmpaxpusegz.c:4:     for (int i = 0; i < n; i++) {
	movl	-12(%rbp), %eax	# i, tmp132
	cmpl	-44(%rbp), %eax	# n, tmp132
	jl	.L7	#,
# /tmp/tmpaxpusegz.c:12:     return 0.0;
	pxor	%xmm0, %xmm0	# _40
# /tmp/tmpaxpusegz.c:13: }
	popq	%rbp	#
	.cfi_def_cfa 7, 8
	ret	
	.cfi_endproc
.LFE0:
	.size	matrix_multiply_asm, .-matrix_multiply_asm
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
