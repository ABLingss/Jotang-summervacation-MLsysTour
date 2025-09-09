	.text
	.file	"tmpjs_gyo76.c"
	.globl	matrix_multiply_asm             # -- Begin function matrix_multiply_asm
	.p2align	4, 0x90
	.type	matrix_multiply_asm,@function
matrix_multiply_asm:                    # @matrix_multiply_asm
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB0_8
# %bb.1:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movl	%ecx, %eax
	leaq	(,%rax,8), %r8
	xorl	%r9d, %r9d
	xorl	%r10d, %r10d
	.p2align	4, 0x90
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_4 Depth 3
	movl	%r9d, %r11d
	leaq	(%rdi,%r11,8), %r11
	movq	%r10, %rbx
	imulq	%rax, %rbx
	leaq	(%rdx,%rbx,8), %rbx
	movq	%rsi, %r14
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_4 Depth 3
	movq	$0, (%rbx,%r15,8)
	xorpd	%xmm0, %xmm0
	movq	%r14, %r12
	xorl	%r13d, %r13d
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	(%r11,%r13,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	(%r12), %xmm1
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%rbx,%r15,8)
	incq	%r13
	addq	%r8, %r12
	cmpq	%r13, %rax
	jne	.LBB0_4
# %bb.5:                                #   in Loop: Header=BB0_3 Depth=2
	incq	%r15
	addq	$8, %r14
	cmpq	%rax, %r15
	jne	.LBB0_3
# %bb.6:                                #   in Loop: Header=BB0_2 Depth=1
	incq	%r10
	addl	%ecx, %r9d
	cmpq	%rax, %r10
	jne	.LBB0_2
# %bb.7:
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
.LBB0_8:
	xorpd	%xmm0, %xmm0
	retq
.Lfunc_end0:
	.size	matrix_multiply_asm, .Lfunc_end0-matrix_multiply_asm
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
