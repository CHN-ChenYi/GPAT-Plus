; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=riscv32 -mattr=+f,+d,+zfh -target-abi=ilp32d | \
; RUN:   FileCheck %s --check-prefixes=RV32
; RUN: llc < %s -mtriple=riscv64 -mattr=+f,+d,+zfh -target-abi=lp64d | \
; RUN:   FileCheck %s --check-prefixes=RV64

; Make sure MachineCSE can combine the adds with the operands commuted.

define void @commute_add_i32(i32 signext %x, i32 signext %y, i32* %p1, i32* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_add_i32:
; RV32:       # %bb.0:
; RV32-NEXT:    add a0, a0, a1
; RV32-NEXT:    sw a0, 0(a2)
; RV32-NEXT:    beqz a4, .LBB0_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    sw a0, 0(a2)
; RV32-NEXT:  .LBB0_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_add_i32:
; RV64:       # %bb.0:
; RV64-NEXT:    addw a0, a0, a1
; RV64-NEXT:    sw a0, 0(a2)
; RV64-NEXT:    beqz a4, .LBB0_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    sw a0, 0(a2)
; RV64-NEXT:  .LBB0_2: # %falseblock
; RV64-NEXT:    ret
  %a = add i32 %x, %y
  store i32 %a, i32* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %b = add i32 %y, %x
  store i32 %b, i32* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_add_i64(i64 %x, i64 %y, i64* %p1, i64* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_add_i64:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a1, a3
; RV32-NEXT:    add a3, a0, a2
; RV32-NEXT:    sltu a0, a3, a0
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    sw a3, 0(a4)
; RV32-NEXT:    sw a0, 4(a4)
; RV32-NEXT:    beqz a6, .LBB1_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    sltu a0, a3, a2
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    sw a3, 0(a4)
; RV32-NEXT:    sw a0, 4(a4)
; RV32-NEXT:  .LBB1_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_add_i64:
; RV64:       # %bb.0:
; RV64-NEXT:    add a0, a0, a1
; RV64-NEXT:    sd a0, 0(a2)
; RV64-NEXT:    beqz a4, .LBB1_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    sd a0, 0(a2)
; RV64-NEXT:  .LBB1_2: # %falseblock
; RV64-NEXT:    ret
  %a = add i64 %x, %y
  store i64 %a, i64* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %b = add i64 %y, %x
  store i64 %b, i64* %p1
  br label %falseblock

falseblock:
  ret void
}

declare half @llvm.fma.f16(half, half, half)

define void @commute_fmadd_f16(half %x, half %y, half %z, half* %p1, half* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmadd_f16:
; RV32:       # %bb.0:
; RV32-NEXT:    fmadd.h ft0, fa0, fa1, fa2
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB2_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:  .LBB2_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmadd_f16:
; RV64:       # %bb.0:
; RV64-NEXT:    fmadd.h ft0, fa0, fa1, fa2
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB2_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:  .LBB2_2: # %falseblock
; RV64-NEXT:    ret
  %a = call half @llvm.fma.f16(half %x, half %y, half %z)
  store half %a, half* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %b = call half @llvm.fma.f16(half %y, half %x, half %z)
  store half %b, half* %p1
  br label %falseblock

falseblock:
  ret void
}

declare float @llvm.fma.f32(float, float, float)

define void @commute_fmadd_f32(float %x, float %y, float %z, float* %p1, float* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmadd_f32:
; RV32:       # %bb.0:
; RV32-NEXT:    fmadd.s ft0, fa0, fa1, fa2
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB3_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:  .LBB3_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmadd_f32:
; RV64:       # %bb.0:
; RV64-NEXT:    fmadd.s ft0, fa0, fa1, fa2
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB3_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:  .LBB3_2: # %falseblock
; RV64-NEXT:    ret
  %a = call float @llvm.fma.f32(float %x, float %y, float %z)
  store float %a, float* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %b = call float @llvm.fma.f32(float %y, float %x, float %z)
  store float %b, float* %p1
  br label %falseblock

falseblock:
  ret void
}

declare double @llvm.fma.f64(double, double, double)

define void @commute_fmadd_f64(double %x, double %y, double %z, double* %p1, double* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmadd_f64:
; RV32:       # %bb.0:
; RV32-NEXT:    fmadd.d ft0, fa0, fa1, fa2
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB4_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:  .LBB4_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmadd_f64:
; RV64:       # %bb.0:
; RV64-NEXT:    fmadd.d ft0, fa0, fa1, fa2
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB4_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:  .LBB4_2: # %falseblock
; RV64-NEXT:    ret
  %a = call double @llvm.fma.f64(double %x, double %y, double %z)
  store double %a, double* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %b = call double @llvm.fma.f64(double %y, double %x, double %z)
  store double %b, double* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fmsub_f16(half %x, half %y, half %z, half* %p1, half* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmsub_f16:
; RV32:       # %bb.0:
; RV32-NEXT:    fmsub.h ft0, fa0, fa1, fa2
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB5_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:  .LBB5_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmsub_f16:
; RV64:       # %bb.0:
; RV64-NEXT:    fmsub.h ft0, fa0, fa1, fa2
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB5_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:  .LBB5_2: # %falseblock
; RV64-NEXT:    ret
  %negz = fneg half %z
  %a = call half @llvm.fma.f16(half %x, half %y, half %negz)
  store half %a, half* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negz2 = fneg half %z
  %b = call half @llvm.fma.f16(half %y, half %x, half %negz2)
  store half %b, half* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fmsub_f32(float %x, float %y, float %z, float* %p1, float* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmsub_f32:
; RV32:       # %bb.0:
; RV32-NEXT:    fmsub.s ft0, fa0, fa1, fa2
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB6_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:  .LBB6_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmsub_f32:
; RV64:       # %bb.0:
; RV64-NEXT:    fmsub.s ft0, fa0, fa1, fa2
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB6_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:  .LBB6_2: # %falseblock
; RV64-NEXT:    ret
  %negz = fneg float %z
  %a = call float @llvm.fma.f32(float %x, float %y, float %negz)
  store float %a, float* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negz2 = fneg float %z
  %b = call float @llvm.fma.f32(float %y, float %x, float %negz2)
  store float %b, float* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fmsub_f64(double %x, double %y, double %z, double* %p1, double* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fmsub_f64:
; RV32:       # %bb.0:
; RV32-NEXT:    fmsub.d ft0, fa0, fa1, fa2
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB7_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:  .LBB7_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fmsub_f64:
; RV64:       # %bb.0:
; RV64-NEXT:    fmsub.d ft0, fa0, fa1, fa2
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB7_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:  .LBB7_2: # %falseblock
; RV64-NEXT:    ret
  %negz = fneg double %z
  %a = call double @llvm.fma.f64(double %x, double %y, double %negz)
  store double %a, double* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negz2 = fneg double %z
  %b = call double @llvm.fma.f64(double %y, double %x, double %negz2)
  store double %b, double* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmadd_f16(half %x, half %y, half %z, half* %p1, half* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmadd_f16:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmadd.h ft0, fa0, fa1, fa2
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB8_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:  .LBB8_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmadd_f16:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmadd.h ft0, fa0, fa1, fa2
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB8_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:  .LBB8_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg half %x
  %negz = fneg half %z
  %a = call half @llvm.fma.f16(half %negx, half %y, half %negz)
  store half %a, half* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg half %y
  %negz2 = fneg half %z
  %b = call half @llvm.fma.f16(half %negy, half %x, half %negz2)
  store half %b, half* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmadd_f32(float %x, float %y, float %z, float* %p1, float* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmadd_f32:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmadd.s ft0, fa0, fa1, fa2
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB9_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:  .LBB9_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmadd_f32:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmadd.s ft0, fa0, fa1, fa2
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB9_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:  .LBB9_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg float %x
  %negz = fneg float %z
  %a = call float @llvm.fma.f32(float %negx, float %y, float %negz)
  store float %a, float* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg float %y
  %negz2 = fneg float %z
  %b = call float @llvm.fma.f32(float %negy, float %x, float %negz2)
  store float %b, float* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmadd_f64(double %x, double %y, double %z, double* %p1, double* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmadd_f64:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmadd.d ft0, fa0, fa1, fa2
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB10_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:  .LBB10_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmadd_f64:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmadd.d ft0, fa0, fa1, fa2
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB10_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:  .LBB10_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg double %x
  %negz = fneg double %z
  %a = call double @llvm.fma.f64(double %negx, double %y, double %negz)
  store double %a, double* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg double %y
  %negz2 = fneg double %z
  %b = call double @llvm.fma.f64(double %negy, double %x, double %negz2)
  store double %b, double* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmsub_f16(half %x, half %y, half %z, half* %p1, half* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmsub_f16:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmsub.h ft0, fa0, fa1, fa2
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB11_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsh ft0, 0(a0)
; RV32-NEXT:  .LBB11_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmsub_f16:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmsub.h ft0, fa0, fa1, fa2
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB11_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsh ft0, 0(a0)
; RV64-NEXT:  .LBB11_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg half %x
  %a = call half @llvm.fma.f16(half %negx, half %y, half %z)
  store half %a, half* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg half %y
  %b = call half @llvm.fma.f16(half %negy, half %x, half %z)
  store half %b, half* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmsub_f32(float %x, float %y, float %z, float* %p1, float* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmsub_f32:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmsub.s ft0, fa0, fa1, fa2
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB12_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsw ft0, 0(a0)
; RV32-NEXT:  .LBB12_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmsub_f32:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmsub.s ft0, fa0, fa1, fa2
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB12_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsw ft0, 0(a0)
; RV64-NEXT:  .LBB12_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg float %x
  %a = call float @llvm.fma.f32(float %negx, float %y, float %z)
  store float %a, float* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg float %y
  %b = call float @llvm.fma.f32(float %negy, float %x, float %z)
  store float %b, float* %p1
  br label %falseblock

falseblock:
  ret void
}

define void @commute_fnmsub_f64(double %x, double %y, double %z, double* %p1, double* %p2, i1 zeroext %cond) {
; RV32-LABEL: commute_fnmsub_f64:
; RV32:       # %bb.0:
; RV32-NEXT:    fnmsub.d ft0, fa0, fa1, fa2
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:    beqz a2, .LBB13_2
; RV32-NEXT:  # %bb.1: # %trueblock
; RV32-NEXT:    fsd ft0, 0(a0)
; RV32-NEXT:  .LBB13_2: # %falseblock
; RV32-NEXT:    ret
;
; RV64-LABEL: commute_fnmsub_f64:
; RV64:       # %bb.0:
; RV64-NEXT:    fnmsub.d ft0, fa0, fa1, fa2
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:    beqz a2, .LBB13_2
; RV64-NEXT:  # %bb.1: # %trueblock
; RV64-NEXT:    fsd ft0, 0(a0)
; RV64-NEXT:  .LBB13_2: # %falseblock
; RV64-NEXT:    ret
  %negx = fneg double %x
  %a = call double @llvm.fma.f64(double %negx, double %y, double %z)
  store double %a, double* %p1
  br i1 %cond, label %trueblock, label %falseblock

trueblock:
  %negy = fneg double %y
  %b = call double @llvm.fma.f64(double %negy, double %x, double %z)
  store double %b, double* %p1
  br label %falseblock

falseblock:
  ret void
}
