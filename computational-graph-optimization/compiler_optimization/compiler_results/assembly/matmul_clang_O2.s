	.text
	.file	"tmp3a2kh68r.c"
	.globl	matrix_multiply_asm             # -- Begin function matrix_multiply_asm
	.p2align	4, 0x90
	.type	matrix_multiply_asm,@function
matrix_multiply_asm:                    # @matrix_multiply_asm
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, -8(%rsp)                  # 8-byte Spill
	movq	%rdi, -16(%rsp)                 # 8-byte Spill
	testl	%ecx, %ecx
	jle	.LBB0_9
# %bb.1:
	movq	%rsi, %rdx
	movl	%ecx, %eax
	movl	%eax, %r8d
	andl	$2147483646, %r8d               # imm = 0x7FFFFFFE
	movq	-16(%rsp), %rsi                 # 8-byte Reload
	leaq	8(%rsi), %rdi
	movq	%rax, %r10
	shlq	$4, %r10
	xorl	%r11d, %r11d
	xorl	%ebx, %ebx
	jmp	.LBB0_2
	.p2align	4, 0x90
.LBB0_8:                                #   in Loop: Header=BB0_2 Depth=1
	incq	%rbx
	addl	%ecx, %r11d
	cmpq	%rax, %rbx
	je	.LBB0_9
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_11 Depth 3
	movl	%r11d, %esi
	leaq	(%rdi,%rsi,8), %r14
	movq	%rbx, %rsi
	imulq	%rax, %rsi
	movl	%esi, %r9d
	movq	-8(%rsp), %r15                  # 8-byte Reload
	leaq	(%r15,%rsi,8), %r15
	movq	-16(%rsp), %rsi                 # 8-byte Reload
	leaq	(%rsi,%r9,8), %r12
	movq	%rdx, %rsi
	xorl	%ebp, %ebp
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=2
	incq	%rbp
	addq	$8, %rsi
	cmpq	%rax, %rbp
	je	.LBB0_8
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_11 Depth 3
	movq	$0, (%r15,%rbp,8)
	xorpd	%xmm0, %xmm0
	cmpl	$1, %ecx
	jne	.LBB0_10
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	xorl	%r9d, %r9d
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_10:                               #   in Loop: Header=BB0_3 Depth=2
	movq	%rsi, %r13
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB0_11:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	-8(%r14,%r9,8), %xmm1           # xmm1 = mem[0],zero
	mulsd	(%r13), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, (%r15,%rbp,8)
	movsd	(%r14,%r9,8), %xmm0             # xmm0 = mem[0],zero
	mulsd	(%r13,%rax,8), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%r15,%rbp,8)
	addq	$2, %r9
	addq	%r10, %r13
	cmpq	%r9, %r8
	jne	.LBB0_11
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	testb	$1, %al
	je	.LBB0_7
# %bb.6:                                #   in Loop: Header=BB0_3 Depth=2
	leaq	(%rdx,%rbp,8), %r13
	movsd	(%r12,%r9,8), %xmm1             # xmm1 = mem[0],zero
	imulq	%rax, %r9
	mulsd	(%r13,%r9,8), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, (%r15,%rbp,8)
	jmp	.LBB0_7
.LBB0_9:
	xorpd	%xmm0, %xmm0
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	matrix_multiply_asm, .Lfunc_end0-matrix_multiply_asm
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
