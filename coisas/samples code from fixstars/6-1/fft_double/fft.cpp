001: #include <stdio.h>
002: #include <stdlib.h>
003: #include <math.h>
004:
005: #ifdef __APPLE__
006: #include <OpenCL/opencl.h>
007: #else
008: #include <CL/cl.h>
009: #endif
010:
011: #include "pgm.h"
012:
013: #define PI 3.14159265358979
014:
015: #define MAX_SOURCE_SIZE (0x100000)
016:
017: #define AMP(a, b) (sqrt((a)*(a)+(b)*(b)))
018:
019: cl_device_id device_id = NULL;
020: cl_context context = NULL;
021: cl_command_queue queue = NULL;
022: cl_program program = NULL;
023:
024: enum Mode {
025:    forward = 0,
026:    inverse = 1
027: };
028:
029: int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
030: {
031:    switch(y) {
032:        case 1:
033:            gws[0] = x;
034:            gws[1] = 1;
035:            lws[0] = 1;
036:            lws[1] = 1;
037:            break;
038:        default:
039:            gws[0] = x;
040:            gws[1] = y;
041:            lws[0] = 1;
042:            lws[1] = 1;
043:            break;
044:    }
045:
046:    return 0;
047: }
048:
049: int fftCore(cl_mem dst, cl_mem src, cl_mem spin, cl_int m, enum Mode direction)
050: {
051:    cl_int ret;
052:
053:    cl_int iter;
054:    cl_uint flag;
055:
056:    cl_int n = 1<<m;
057:
058:    cl_event kernelDone;
059:
060:    cl_kernel brev = NULL;
061:    cl_kernel bfly = NULL;
062:    cl_kernel norm = NULL;
063:
064:    brev = clCreateKernel(program, "bitReverse", &ret);
065:    bfly = clCreateKernel(program, "butterfly", &ret);
066:    norm = clCreateKernel(program, "norm", &ret);
067:
068:    size_t gws[2];
069:    size_t lws[2];
070:
071:    switch (direction) {
072:        case forward:flag = 0x00000000; break;
073:        case inverse:flag = 0x80000000; break;
074:    }
075:
076:    ret = clSetKernelArg(brev, 0, sizeof(cl_mem), (void *)&dst);
077:    ret = clSetKernelArg(brev, 1, sizeof(cl_mem), (void *)&src);
078:    ret = clSetKernelArg(brev, 2, sizeof(cl_int), (void *)&m);
079:    ret = clSetKernelArg(brev, 3, sizeof(cl_int), (void *)&n);
080:
081:    ret = clSetKernelArg(bfly, 0, sizeof(cl_mem), (void *)&dst);
082:    ret = clSetKernelArg(bfly, 1, sizeof(cl_mem), (void *)&spin);
083:    ret = clSetKernelArg(bfly, 2, sizeof(cl_int), (void *)&m);
084:    ret = clSetKernelArg(bfly, 3, sizeof(cl_int), (void *)&n);
085:    ret = clSetKernelArg(bfly, 5, sizeof(cl_uint), (void *)&flag);
086:
087:    ret = clSetKernelArg(norm, 0, sizeof(cl_mem), (void *)&dst);
088:    ret = clSetKernelArg(norm, 1, sizeof(cl_int), (void *)&n);
089:
090:    /* Reverse bit ordering */
091:    setWorkSize(gws, lws, n, n);
092:    ret = clEnqueueNDRangeKernel(queue, brev, 2, NULL, gws, lws, 0, NULL, NULL);
093:
094:    /* Perform Butterfly Operations*/
095:    setWorkSize(gws, lws, n/2, n);
096:    for (iter=1; iter <= m; iter++){
097:        ret = clSetKernelArg(bfly, 4, sizeof(cl_int), (void *)&iter);
098:        ret = clEnqueueNDRangeKernel(queue, bfly, 2, NULL, gws, lws, 0, NULL, &kernelDone);
099:        ret = clWaitForEvents(1, &kernelDone);
100:    }
101:
102:    if (direction == inverse) {
103:        setWorkSize(gws, lws, n, n);
104:        ret = clEnqueueNDRangeKernel(queue, norm, 2, NULL, gws, lws, 0, NULL, &kernelDone);
105:        ret = clWaitForEvents(1, &kernelDone);
106:    }
107:
108:    ret = clReleaseKernel(bfly);
109:    ret = clReleaseKernel(brev);
110:    ret = clReleaseKernel(norm);
111:
112:    return 0;
113: }
114:
115: int main()
116: {
117:    cl_mem xmobj = NULL;
118:    cl_mem rmobj = NULL;
119:    cl_mem wmobj = NULL;
120:    cl_kernel sfac = NULL;
121:    cl_kernel trns = NULL;
122:    cl_kernel hpfl = NULL;
123:
124:    cl_platform_id platform_id = NULL;
125:
126:    cl_uint ret_num_devices;
127:    cl_uint ret_num_platforms;
128:
129:    cl_int ret;
130:
131:    cl_float2 *xm;
132:    cl_float2 *rm;
133:    cl_float2 *wm;
134:
135:    pgm_t ipgm;
136:    pgm_t opgm;
137:
138:    FILE *fp;
139:    const char fileName[] = "./fft.cl";
140:    size_t source_size;
141:    char *source_str;
142:    cl_int i, j;
143:    cl_int n;
144:    cl_int m;
145:
146:    size_t gws[2];
147:    size_t lws[2];
148:
149:    /* Load kernel source code */
150:    fp = fopen(fileName, "r");
151:    if (!fp) {
152:        fprintf(stderr, "Failed to load kernel.\n");
153:        exit(1);
154:    }
155:    source_str = (char *)malloc(MAX_SOURCE_SIZE);
156:    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
157:    fclose( fp );
158:
159:    /* Read image */
160:    readPGM(&ipgm, "lena.pgm");
161:
162:    n = ipgm.width;
163:    m = (cl_int)(log((double)n)/log(2.0));
164:
165:    xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
166:    rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
167:    wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));
168:
169:    for (i=0; i < n; i++) {
170:        for (j=0; j < n; j++) {
171:            ((float*)xm)[(2*n*j)+2*i+0] = (float)ipgm.buf[n*j+i];
172:            ((float*)xm)[(2*n*j)+2*i+1] = (float)0;
173:        }
174:    }
175:
176:    /* Get platform/device  */
177:    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
178:    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
179:
180:    /* Create OpenCL context */
181:    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
182:
183:    /* Create Command queue */
184:    queue = clCreateCommandQueue(context, device_id, 0, &ret);
185:
186:    /* Create Buffer Objects */
187:    xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
188:    rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
189:    wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, (n/2)*sizeof(cl_float2), NULL, &ret);
190:
191:    /* Transfer data to memory buffer */
192:    ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);
193:
194:    /* Create kernel program from source */
195:    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
196:
197:    /* Build kernel program */
198:    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
199:
200:    /* Create OpenCL Kernel */
201:    sfac = clCreateKernel(program, "spinFact", &ret);
202:    trns = clCreateKernel(program, "transpose", &ret);
203:    hpfl = clCreateKernel(program, "highPassFilter", &ret);
204:
205:    /* Create spin factor */
206:    ret = clSetKernelArg(sfac, 0, sizeof(cl_mem), (void *)&wmobj);
207:    ret = clSetKernelArg(sfac, 1, sizeof(cl_int), (void *)&n);
208:    setWorkSize(gws, lws, n/2, 1);
209:    ret = clEnqueueNDRangeKernel(queue, sfac, 1, NULL, gws, lws, 0, NULL, NULL);
210:
211:    /* Butterfly Operation */
212:    fftCore(rmobj, xmobj, wmobj, m, forward);
213:
214:    /* Transpose matrix */
215:    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&xmobj);
216:    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&rmobj);
217:    ret = clSetKernelArg(trns, 2, sizeof(cl_int), (void *)&n);
218:    setWorkSize(gws, lws, n, n);
219:    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, NULL);
220:
221:    /* Butterfly Operation */
222:    fftCore(rmobj, xmobj, wmobj, m, forward);
223:
224:    /* Apply high-pass filter */
225:    cl_int radius = n/8;
226:    ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&rmobj);
227:    ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
228:    ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
229:    setWorkSize(gws, lws, n, n);
230:    ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, NULL);
231:
232:    /* Inverse FFT */
233:
234:    /* Butterfly Operation */
235:    fftCore(xmobj, rmobj, wmobj, m, inverse);
236:
237:    /* Transpose matrix */
238:    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&rmobj);
239:    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&xmobj);
240:    setWorkSize(gws, lws, n, n);
241:    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, NULL);
242:
243:    /* Butterfly Operation */
244:    fftCore(xmobj, rmobj, wmobj, m, inverse);
245:
246:    /* Read data from memory buffer */
247:    ret = clEnqueueReadBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);
248:
249:    /* Calculate amplitude */
250:    float* ampd;
251:    ampd = (float*)malloc(n*n*sizeof(float));
252:    for (i=0; i < n; i++) {
253:        for (j=0; j < n; j++) {
254:            ampd[n*((i))+((j))] = (AMP(((float*)xm)[(2*n*i)+2*j], ((float*)xm)[(2*n*i)+2*j+1]));
255:        }
256:    }
257:    opgm.width = n;
258:    opgm.height = n;
259:    normalizeF2PGM(&opgm, ampd);
260:    free(ampd);
261:
262:    /* Write out image */
263:    writePGM(&opgm, "output.pgm");
264:
265:    /* Finalizations*/
266:    ret = clFlush(queue);
267:    ret = clFinish(queue);
268:    ret = clReleaseKernel(hpfl);
269:    ret = clReleaseKernel(trns);
270:    ret = clReleaseKernel(sfac);
271:    ret = clReleaseProgram(program);
272:    ret = clReleaseMemObject(xmobj);
273:    ret = clReleaseMemObject(rmobj);
274:    ret = clReleaseMemObject(wmobj);
275:    ret = clReleaseCommandQueue(queue);
276:    ret = clReleaseContext(context);
277:
278:    destroyPGM(&ipgm);
279:    destroyPGM(&opgm);
280:
281:    free(source_str);
282:    free(wm);
283:    free(rm);
284:    free(xm);
285:
286:    return 0;
287: }