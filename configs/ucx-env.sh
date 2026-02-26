# UCX env settings (tune per hardware/driver)
export UCX_TLS=rc,ud,mm,self,sm,cuda_copy,cuda_ipc
export UCX_MAX_RNDV_RAILS=2
export UCX_RNDV_THRESH=8192
export UCX_MEMTYPE_CACHE=n
