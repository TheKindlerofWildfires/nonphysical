/*
    To use constant memory it needs to be defined at the top of the file like
        .const .align $data_type_size .b8 $name[size]  
    Then it's loaded like
    ld.const.$data_type $register [name+offset]

    I can def constant slice to be a read only (ptx side) and just normally writeable with cu_slice rules (via memcpy so symbol)
    however there's the small issue of getting the .const to compile in correctly
    also copying data to the symbol is a little more fun, I think I use cucopytodevice and I need a symbol pointer

    maybe 
    float *dcoeffs;
cudaGetSymbolAddress((void **)&dcoeffs, coeffs);
cudaMemcpy(dcoeffs, hostData, 8*sizeof(float), cudaMemcpyHostToDevice);

except more like
cuGetProcAddress_v2(feel like this gets functions)
ffi::cuMemcpyHtoD_v2(

this is going to be very painful around moving to const memory, lets do shared mem instead

*/