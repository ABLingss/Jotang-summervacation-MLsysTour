	.text
	.file	"tmp6ag_shb6.c"
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
	movq	%rdx, -32(%rsp)                 # 8-byte Spill
	movq	%rsi, -96(%rsp)                 # 8-byte Spill
	movq	%rdi, -104(%rsp)                # 8-byte Spill
	testl	%ecx, %ecx
	jle	.LBB0_45
# %bb.1:
	movl	%ecx, %eax
	movl	%eax, %edx
	andl	$2147483644, %edx               # imm = 0x7FFFFFFC
	leaq	(,%rax,8), %r15
	movl	%eax, %ebp
	movl	%eax, %r10d
	movl	%eax, %r11d
	movq	%rax, %r14
	andl	$3, %ebp
	andl	$7, %r10d
	andl	$2147483640, %r11d              # imm = 0x7FFFFFF8
	shlq	$6, %r14
	xorl	%edi, %edi
	movl	%ecx, -116(%rsp)                # 4-byte Spill
	movq	%rdx, -8(%rsp)                  # 8-byte Spill
	movq	-104(%rsp), %rdx                # 8-byte Reload
	movq	%rbp, -40(%rsp)                 # 8-byte Spill
	addq	$56, %rdx
	movq	%rdx, -48(%rsp)                 # 8-byte Spill
	movq	-96(%rsp), %rdx                 # 8-byte Reload
	leaq	8(%rdx), %rsi
	leaq	16(%rdx), %r8
	addq	$24, %rdx
	movq	%rdx, -72(%rsp)                 # 8-byte Spill
	xorl	%edx, %edx
	movq	%rsi, -56(%rsp)                 # 8-byte Spill
	movq	%r8, -64(%rsp)                  # 8-byte Spill
	jmp	.LBB0_2
	.p2align	4, 0x90
.LBB0_44:                               #   in Loop: Header=BB0_2 Depth=1
	movq	-16(%rsp), %rdx                 # 8-byte Reload
	movq	-24(%rsp), %rdi                 # 8-byte Reload
	incq	%rdi
	addl	%ecx, %edx
	cmpq	%rax, %rdi
	je	.LBB0_45
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
                                        #       Child Loop BB0_8 Depth 3
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_22 Depth 3
                                        #       Child Loop BB0_25 Depth 3
                                        #       Child Loop BB0_29 Depth 3
                                        #       Child Loop BB0_32 Depth 3
                                        #     Child Loop BB0_36 Depth 2
                                        #       Child Loop BB0_39 Depth 3
                                        #       Child Loop BB0_42 Depth 3
	movl	%edx, %esi
	movq	%rdx, -16(%rsp)                 # 8-byte Spill
	movq	%rdi, %rdx
	movq	%rdi, -24(%rsp)                 # 8-byte Spill
	movq	-48(%rsp), %r9                  # 8-byte Reload
	movq	-32(%rsp), %rdi                 # 8-byte Reload
	imulq	%rax, %rdx
	leaq	(,%rsi,8), %r8
	movq	%r8, -88(%rsp)                  # 8-byte Spill
	leaq	(%r9,%rsi,8), %rbx
	leaq	(%rdi,%rdx,8), %rdx
	cmpl	$4, %ecx
	jae	.LBB0_4
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	xorl	%r8d, %r8d
.LBB0_34:                               #   in Loop: Header=BB0_2 Depth=1
	movq	-40(%rsp), %rbp                 # 8-byte Reload
	testq	%rbp, %rbp
	je	.LBB0_44
# %bb.35:                               #   in Loop: Header=BB0_2 Depth=1
	movq	-96(%rsp), %rsi                 # 8-byte Reload
	movq	-88(%rsp), %rdi                 # 8-byte Reload
	addq	-104(%rsp), %rdi                # 8-byte Folded Reload
	leaq	(%rsi,%r8,8), %rsi
	movq	%rdi, -88(%rsp)                 # 8-byte Spill
	xorl	%edi, %edi
	jmp	.LBB0_36
	.p2align	4, 0x90
.LBB0_43:                               #   in Loop: Header=BB0_36 Depth=2
	incq	%r8
	incq	%rdi
	addq	$8, %rsi
	cmpq	%rbp, %rdi
	je	.LBB0_44
.LBB0_36:                               #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_39 Depth 3
                                        #       Child Loop BB0_42 Depth 3
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	$0, (%rdx,%r8,8)
	cmpl	$8, %ecx
	jae	.LBB0_38
# %bb.37:                               #   in Loop: Header=BB0_36 Depth=2
	xorl	%r9d, %r9d
	jmp	.LBB0_40
	.p2align	4, 0x90
.LBB0_38:                               #   in Loop: Header=BB0_36 Depth=2
	movq	%rsi, %r13
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB0_39:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_36 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	-56(%rbx,%r9,8), %xmm1          # xmm1 = mem[0],zero
	leaq	(%r13,%r15), %r12
	vfmadd132sd	(%r13), %xmm0, %xmm1    # xmm1 = (xmm1 * mem) + xmm0
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-48(%rbx,%r9,8), %xmm0          # xmm0 = mem[0],zero
	vfmadd132sd	(%r13,%r15), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r14, %r13
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-40(%rbx,%r9,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%r12), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %r12
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-32(%rbx,%r9,8), %xmm0          # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%r12), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %r12
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-24(%rbx,%r9,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%r12), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %r12
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-16(%rbx,%r9,8), %xmm0          # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%r12), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %r12
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-8(%rbx,%r9,8), %xmm1           # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%r12), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %r12
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	(%rbx,%r9,8), %xmm0             # xmm0 = mem[0],zero
	addq	$8, %r9
	vfmadd132sd	(%r15,%r12), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	vmovsd	%xmm0, (%rdx,%r8,8)
	cmpq	%r9, %r11
	jne	.LBB0_39
.LBB0_40:                               #   in Loop: Header=BB0_36 Depth=2
	testb	$7, %al
	je	.LBB0_43
# %bb.41:                               #   in Loop: Header=BB0_36 Depth=2
	movq	-88(%rsp), %r12                 # 8-byte Reload
	movq	%r15, %r13
	imulq	%r9, %r13
	addq	%rsi, %r13
	leaq	(%r12,%r9,8), %r9
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_42:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_36 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%r9,%r12,8), %xmm1             # xmm1 = mem[0],zero
	incq	%r12
	vfmadd231sd	(%r13), %xmm1, %xmm0    # xmm0 = (xmm1 * mem) + xmm0
	addq	%r15, %r13
	vmovsd	%xmm0, (%rdx,%r8,8)
	cmpq	%r12, %r10
	jne	.LBB0_42
	jmp	.LBB0_43
	.p2align	4, 0x90
.LBB0_4:                                #   in Loop: Header=BB0_2 Depth=1
	movq	-104(%rsp), %rsi                # 8-byte Reload
	movq	-72(%rsp), %rdi                 # 8-byte Reload
	movq	-64(%rsp), %r13                 # 8-byte Reload
	movq	-96(%rsp), %r9                  # 8-byte Reload
	addq	%r8, %rsi
	movq	%rdi, -80(%rsp)                 # 8-byte Spill
	xorl	%r8d, %r8d
	movq	%rsi, -112(%rsp)                # 8-byte Spill
	movq	-56(%rsp), %rsi                 # 8-byte Reload
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_33:                               #   in Loop: Header=BB0_5 Depth=2
	addq	$32, -80(%rsp)                  # 8-byte Folded Spill
	movl	-116(%rsp), %ecx                # 4-byte Reload
	addq	$4, %r8
	addq	$32, %r9
	addq	$32, %rsi
	addq	$32, %r13
	cmpq	-8(%rsp), %r8                   # 8-byte Folded Reload
	je	.LBB0_34
.LBB0_5:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_22 Depth 3
                                        #       Child Loop BB0_25 Depth 3
                                        #       Child Loop BB0_29 Depth 3
                                        #       Child Loop BB0_32 Depth 3
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	$0, (%rdx,%r8,8)
	cmpl	$8, %ecx
	jae	.LBB0_7
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=2
	xorl	%edi, %edi
	jmp	.LBB0_9
	.p2align	4, 0x90
.LBB0_7:                                #   in Loop: Header=BB0_5 Depth=2
	movq	%r9, %r12
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_8:                                #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	-56(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	leaq	(%r12,%rax,8), %rbp
	vfmadd132sd	(%r12), %xmm0, %xmm1    # xmm1 = (xmm1 * mem) + xmm0
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-48(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r12,%rax,8), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r14, %r12
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-40(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rbp), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rbp
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-32(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rbp), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rbp
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-24(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rbp), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rbp
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	-16(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rbp), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rbp
	vmovsd	%xmm0, (%rdx,%r8,8)
	vmovsd	-8(%rbx,%rdi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rbp), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rbp
	vmovsd	%xmm1, (%rdx,%r8,8)
	vmovsd	(%rbx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	addq	$8, %rdi
	vfmadd132sd	(%r15,%rbp), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	vmovsd	%xmm0, (%rdx,%r8,8)
	cmpq	%rdi, %r11
	jne	.LBB0_8
.LBB0_9:                                #   in Loop: Header=BB0_5 Depth=2
	testq	%r10, %r10
	je	.LBB0_12
# %bb.10:                               #   in Loop: Header=BB0_5 Depth=2
	movq	-112(%rsp), %r12                # 8-byte Reload
	movq	%r15, %rcx
	imulq	%rdi, %rcx
	addq	%r9, %rcx
	leaq	(%r12,%rdi,8), %rdi
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_11:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r12,8), %xmm1            # xmm1 = mem[0],zero
	incq	%r12
	vfmadd231sd	(%rcx), %xmm1, %xmm0    # xmm0 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r8,8)
	cmpq	%r12, %r10
	jne	.LBB0_11
.LBB0_12:                               #   in Loop: Header=BB0_5 Depth=2
	movq	%r8, %r12
	orq	$1, %r12
	cmpl	$8, -116(%rsp)                  # 4-byte Folded Reload
	movq	$0, (%rdx,%r12,8)
	jae	.LBB0_14
# %bb.13:                               #   in Loop: Header=BB0_5 Depth=2
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_14:                               #   in Loop: Header=BB0_5 Depth=2
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%rsi, %rbp
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_15:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	-56(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	leaq	(%rbp,%r15), %rcx
	vfmadd132sd	(%rbp), %xmm0, %xmm1    # xmm1 = (xmm1 * mem) + xmm0
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-48(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%rbp,%r15), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r14, %rbp
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-40(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-32(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-24(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-16(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-8(%rbx,%rdi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	(%rbx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	addq	$8, %rdi
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rdi, %r11
	jne	.LBB0_15
.LBB0_16:                               #   in Loop: Header=BB0_5 Depth=2
	testb	$7, %al
	je	.LBB0_19
# %bb.17:                               #   in Loop: Header=BB0_5 Depth=2
	movq	-112(%rsp), %rbp                # 8-byte Reload
	movq	%r15, %rcx
	imulq	%rdi, %rcx
	addq	%rsi, %rcx
	leaq	(%rbp,%rdi,8), %rdi
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_18:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%rbp,8), %xmm1            # xmm1 = mem[0],zero
	incq	%rbp
	vfmadd231sd	(%rcx), %xmm1, %xmm0    # xmm0 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rbp, %r10
	jne	.LBB0_18
.LBB0_19:                               #   in Loop: Header=BB0_5 Depth=2
	movq	%r8, %r12
	orq	$2, %r12
	cmpl	$8, -116(%rsp)                  # 4-byte Folded Reload
	movq	$0, (%rdx,%r12,8)
	jae	.LBB0_21
# %bb.20:                               #   in Loop: Header=BB0_5 Depth=2
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	jmp	.LBB0_23
	.p2align	4, 0x90
.LBB0_21:                               #   in Loop: Header=BB0_5 Depth=2
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%r13, %rbp
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_22:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	-56(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	leaq	(%rbp,%r15), %rcx
	vfmadd132sd	(%rbp), %xmm0, %xmm1    # xmm1 = (xmm1 * mem) + xmm0
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-48(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%rbp,%r15), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r14, %rbp
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-40(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-32(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-24(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-16(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-8(%rbx,%rdi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	(%rbx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	addq	$8, %rdi
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rdi, %r11
	jne	.LBB0_22
.LBB0_23:                               #   in Loop: Header=BB0_5 Depth=2
	testb	$7, %al
	je	.LBB0_26
# %bb.24:                               #   in Loop: Header=BB0_5 Depth=2
	movq	-112(%rsp), %rbp                # 8-byte Reload
	movq	%r15, %rcx
	imulq	%rdi, %rcx
	addq	%r13, %rcx
	leaq	(%rbp,%rdi,8), %rdi
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_25:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%rbp,8), %xmm1            # xmm1 = mem[0],zero
	incq	%rbp
	vfmadd231sd	(%rcx), %xmm1, %xmm0    # xmm0 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rbp, %r10
	jne	.LBB0_25
.LBB0_26:                               #   in Loop: Header=BB0_5 Depth=2
	movq	%r8, %r12
	orq	$3, %r12
	cmpl	$8, -116(%rsp)                  # 4-byte Folded Reload
	movq	$0, (%rdx,%r12,8)
	jae	.LBB0_28
# %bb.27:                               #   in Loop: Header=BB0_5 Depth=2
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	jmp	.LBB0_30
	.p2align	4, 0x90
.LBB0_28:                               #   in Loop: Header=BB0_5 Depth=2
	movq	-80(%rsp), %rbp                 # 8-byte Reload
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB0_29:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	-56(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	leaq	(%rbp,%r15), %rcx
	vfmadd132sd	(%rbp), %xmm0, %xmm1    # xmm1 = (xmm1 * mem) + xmm0
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-48(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%rbp,%r15), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r14, %rbp
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-40(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-32(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-24(%rbx,%rdi,8), %xmm1         # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	-16(%rbx,%rdi,8), %xmm0         # xmm0 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	vmovsd	-8(%rbx,%rdi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd132sd	(%r15,%rcx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm1, (%rdx,%r12,8)
	vmovsd	(%rbx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	addq	$8, %rdi
	vfmadd132sd	(%r15,%rcx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rdi, %r11
	jne	.LBB0_29
.LBB0_30:                               #   in Loop: Header=BB0_5 Depth=2
	testb	$7, %al
	je	.LBB0_33
# %bb.31:                               #   in Loop: Header=BB0_5 Depth=2
	movq	-112(%rsp), %rbp                # 8-byte Reload
	movq	%r15, %rcx
	imulq	%rdi, %rcx
	addq	-80(%rsp), %rcx                 # 8-byte Folded Reload
	leaq	(%rbp,%rdi,8), %rdi
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_32:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%rbp,8), %xmm1            # xmm1 = mem[0],zero
	incq	%rbp
	vfmadd231sd	(%rcx), %xmm1, %xmm0    # xmm0 = (xmm1 * mem) + xmm0
	addq	%r15, %rcx
	vmovsd	%xmm0, (%rdx,%r12,8)
	cmpq	%rbp, %r10
	jne	.LBB0_32
	jmp	.LBB0_33
.LBB0_45:
	vxorpd	%xmm0, %xmm0, %xmm0
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
