#ifndef incl_HPHP_VM_CODE_GEN_TLS_ARM_H_
#define incl_HPHP_VM_CODE_GEN_TLS_ARM_H_

#include "hphp/runtime/vm/jit/vasm-gen.h"
#include "hphp/runtime/vm/jit/vasm-instr.h"
#include "hphp/runtime/vm/jit/vasm-reg.h"

#include "hphp/runtime/vm/jit/abi-arm.h"
#include "hphp/util/thread-local.h"

namespace HPHP { namespace jit { namespace arm { namespace detail {

///////////////////////////////////////////////////////////////////////////////

/*
 */
template<typename T>
Vptr emitTLSAddr(Vout& v, TLSDatum<T> datum) {
  auto const d = v.makeReg();
  auto const s = ssize_t(uintptr_t(datum.tls));
  v << ldimmq{s, d};
  return d[0];
}

}}}}

#endif
