use crate::cuda::error::CuError::*;
use crate::cuda::ffi::{cudaError_enum, CUresult};
use core::result::Result;

#[derive(Copy, Clone, Debug)]
pub enum CuError {
    CuSuccess,
    CuInvalidValue,
    CuOutOfMemory,
    CuNotInitialized,
    CuDeinitialized,
    CuProfilerDisabled,
    CuProfilerNotInitialized,
    CuProfilerAlreadyStarted,
    CuProfilerAlreadyStopped,
    CuStubLibrary,
    CuDeviceUnavailable,
    CuNoDevice,
    CuInvalidDevice,
    CuDeviceNotLicensed,
    CuInvalidImage,
    CuInvalidContext,
    CuContextAlreadyCurrent,
    CuMapFailed,
    CuUnmapFailed,
    CuArrayIsMapped,
    CuAlreadyMapped,
    CuNoBinaryForGPU,
    CuAlreadyAcquired,
    CuNotMapped,
    CuNotMappedAsArray,
    CuNotMappedAsPointer,
    CuEccUncorrectable,
    CuUnsupportedLimit,
    CuContextAlreadyInUse,
    CuPeerAccessUnsupported,
    CuInvalidPtx,
    CuInvalidGraphicsContext,
    CuNvlinkUncorrectable,
    CuJitCompilerNotFound,
    CuUnsupportedPtxVersion,
    CuJitCompilationDisabled,
    CuUnsupportedExecAffinity,
    CuUnsupportedDevSideSync,
    CuInvalidSource,
    CuFileNotFound,
    CuSharedObjectSymbolNotFound,
    CuSharedObjectInitFailed,
    CuOperatingSystem,
    CuInvalidHandle,
    CuIllegalState,
    CuLossyQuery,
    CuNotFound,
    CuNotReady,
    CuIllegalAddress,
    CuLaunchOutOfResources,
    CuLaunchTimeout,
    CuLaunchIncompatibleTexturing,
    CuPeerAccessAlreadyEnabled,
    CuPeerAccessNotEnabled,
    CuPrimaryContextActive,
    CuContextIsDestroyed,
    CuAssert,
    CuTooManyPeers,
    CuHostMemoryAlreadyRegistered,
    CuHostMemoryNotRegistered,
    CuHardwareStackError,
    CuIllegalInstruction,
    CuMisalignedAddress,
    CuInvalidAddressSpace,
    CuInvalidPc,
    CuLaunchFailed,
    CuCooperativeLaunchTooLarge,
    CuNotPermitted,
    CuNotSupported,
    CuSystemNotReady,
    CuSystemDriverMismatch,
    CuCompatNotSupportedOnDevice,
    CuMpsConnectionFailed,
    CuMpsRpcFailure,
    CuMpsServerNotReady,
    CuMpsMaxClientsReached,
    CuMpsMaxConnectionsReached,
    CuMpsClientTerminated,
    CuCdpNotSupported,
    CuCdpVersionMismatch,
    CuStreamCaptureUnsupported,
    CuStreamCaptureInvalidated,
    CuStreamCaptureMerge,
    CuStreamCaptureUnmatched,
    CuStreamCaptureUnjoined,
    CuStreamCaptureIsolation,
    CuStreamCaptureImplicit,
    CuCapturedEvent,
    CuStreamCaptureWrongThread,
    CuTimeout,
    CuGraphExecUpdateFailure,
    CuExternalDevice,
    CuInvalidClusterSize,
    CuUnknown,
}

impl CuError {
    pub fn check(result: CUresult) -> Result<(), CuError> {
        match result {
            cudaError_enum::CUDA_SUCCESS => Ok(()),
            cudaError_enum::CUDA_ERROR_INVALID_VALUE => Err(CuInvalidValue),
            cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => Err(CuOutOfMemory),
            cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => Err(CuNotInitialized),
            cudaError_enum::CUDA_ERROR_DEINITIALIZED => Err(CuDeinitialized),
            cudaError_enum::CUDA_ERROR_PROFILER_DISABLED => Err(CuProfilerDisabled),
            cudaError_enum::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Err(CuProfilerNotInitialized),
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STARTED => Err(CuProfilerAlreadyStarted),
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Err(CuProfilerAlreadyStopped),
            cudaError_enum::CUDA_ERROR_STUB_LIBRARY => Err(CuStubLibrary),
            cudaError_enum::CUDA_ERROR_DEVICE_UNAVAILABLE => Err(CuDeviceUnavailable),
            cudaError_enum::CUDA_ERROR_NO_DEVICE => Err(CuNoDevice),
            cudaError_enum::CUDA_ERROR_INVALID_DEVICE => Err(CuInvalidDevice),
            cudaError_enum::CUDA_ERROR_DEVICE_NOT_LICENSED => Err(CuDeviceNotLicensed),
            cudaError_enum::CUDA_ERROR_INVALID_IMAGE => Err(CuInvalidImage),
            cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => Err(CuInvalidContext),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(CuContextAlreadyCurrent),
            cudaError_enum::CUDA_ERROR_MAP_FAILED => Err(CuMapFailed),
            cudaError_enum::CUDA_ERROR_UNMAP_FAILED => Err(CuUnmapFailed),
            cudaError_enum::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CuArrayIsMapped),
            cudaError_enum::CUDA_ERROR_ALREADY_MAPPED => Err(CuAlreadyMapped),
            cudaError_enum::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CuNoBinaryForGPU),
            cudaError_enum::CUDA_ERROR_ALREADY_ACQUIRED => Err(CuAlreadyAcquired),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED => Err(CuNotMapped),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CuNotMappedAsArray),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CuNotMappedAsPointer),
            cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CuEccUncorrectable),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CuUnsupportedLimit),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CuContextAlreadyInUse),
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(CuPeerAccessUnsupported),
            cudaError_enum::CUDA_ERROR_INVALID_PTX => Err(CuInvalidPtx),
            cudaError_enum::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Err(CuInvalidGraphicsContext),
            cudaError_enum::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CuNvlinkUncorrectable),
            cudaError_enum::CUDA_ERROR_JIT_COMPILER_NOT_FOUND => Err(CuJitCompilerNotFound),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => Err(CuUnsupportedPtxVersion),
            cudaError_enum::CUDA_ERROR_JIT_COMPILATION_DISABLED => Err(CuJitCompilationDisabled),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => Err(CuUnsupportedExecAffinity),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC => Err(CuUnsupportedDevSideSync),
            cudaError_enum::CUDA_ERROR_INVALID_SOURCE => Err(CuInvalidSource),
            cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND => Err(CuFileNotFound),
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CuSharedObjectSymbolNotFound)
            }
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Err(CuSharedObjectInitFailed),
            cudaError_enum::CUDA_ERROR_OPERATING_SYSTEM => Err(CuOperatingSystem),
            cudaError_enum::CUDA_ERROR_INVALID_HANDLE => Err(CuInvalidHandle),
            cudaError_enum::CUDA_ERROR_ILLEGAL_STATE => Err(CuIllegalState),
            cudaError_enum::CUDA_ERROR_LOSSY_QUERY => Err(CuLossyQuery),
            cudaError_enum::CUDA_ERROR_NOT_FOUND => Err(CuNotFound),
            cudaError_enum::CUDA_ERROR_NOT_READY => Err(CuNotReady),
            cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CuIllegalAddress),
            cudaError_enum::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CuLaunchOutOfResources),
            cudaError_enum::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CuLaunchTimeout),
            cudaError_enum::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CuLaunchIncompatibleTexturing)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CuPeerAccessAlreadyEnabled)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CuPeerAccessNotEnabled),
            cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CuPrimaryContextActive),
            cudaError_enum::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CuContextIsDestroyed),
            cudaError_enum::CUDA_ERROR_ASSERT => Err(CuAssert),
            cudaError_enum::CUDA_ERROR_TOO_MANY_PEERS => Err(CuTooManyPeers),
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CuHostMemoryAlreadyRegistered)
            }
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Err(CuHostMemoryNotRegistered),
            cudaError_enum::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CuHardwareStackError),
            cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CuIllegalInstruction),
            cudaError_enum::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CuMisalignedAddress),
            cudaError_enum::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CuInvalidAddressSpace),
            cudaError_enum::CUDA_ERROR_INVALID_PC => Err(CuInvalidPc),
            cudaError_enum::CUDA_ERROR_LAUNCH_FAILED => Err(CuLaunchFailed),
            cudaError_enum::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => {
                Err(CuCooperativeLaunchTooLarge)
            }
            cudaError_enum::CUDA_ERROR_NOT_PERMITTED => Err(CuNotPermitted),
            cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Err(CuNotSupported),
            cudaError_enum::CUDA_ERROR_SYSTEM_NOT_READY => Err(CuSystemNotReady),
            cudaError_enum::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => Err(CuSystemDriverMismatch),
            cudaError_enum::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => {
                Err(CuCompatNotSupportedOnDevice)
            }
            cudaError_enum::CUDA_ERROR_MPS_CONNECTION_FAILED => Err(CuMpsConnectionFailed),
            cudaError_enum::CUDA_ERROR_MPS_RPC_FAILURE => Err(CuMpsRpcFailure),
            cudaError_enum::CUDA_ERROR_MPS_SERVER_NOT_READY => Err(CuMpsServerNotReady),
            cudaError_enum::CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => Err(CuMpsMaxClientsReached),
            cudaError_enum::CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => {
                Err(CuMpsMaxConnectionsReached)
            }
            cudaError_enum::CUDA_ERROR_MPS_CLIENT_TERMINATED => Err(CuMpsClientTerminated),
            cudaError_enum::CUDA_ERROR_CDP_NOT_SUPPORTED => Err(CuCdpNotSupported),
            cudaError_enum::CUDA_ERROR_CDP_VERSION_MISMATCH => Err(CuCdpVersionMismatch),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => {
                Err(CuStreamCaptureUnsupported)
            }
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => {
                Err(CuStreamCaptureInvalidated)
            }
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_MERGE => Err(CuStreamCaptureMerge),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => Err(CuStreamCaptureUnmatched),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNJOINED => Err(CuStreamCaptureUnjoined),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_ISOLATION => Err(CuStreamCaptureIsolation),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => Err(CuStreamCaptureImplicit),
            cudaError_enum::CUDA_ERROR_CAPTURED_EVENT => Err(CuCapturedEvent),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => {
                Err(CuStreamCaptureWrongThread)
            }
            cudaError_enum::CUDA_ERROR_TIMEOUT => Err(CuTimeout),
            cudaError_enum::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => Err(CuGraphExecUpdateFailure),
            cudaError_enum::CUDA_ERROR_EXTERNAL_DEVICE => Err(CuExternalDevice),
            cudaError_enum::CUDA_ERROR_INVALID_CLUSTER_SIZE => Err(CuInvalidClusterSize),
            cudaError_enum::CUDA_ERROR_UNKNOWN => Err(CuUnknown),
        }
    }
}
