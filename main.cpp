#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <ctime>

#ifndef MIN
#define MIN(A,B)	(((A) < (B)) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A,B)	(((A) > (B)) ? (A) : (B))
#endif

using namespace std;

typedef struct _Sample
{
    double *data;//之所以用double是为了后面计算结果更精确
    double *label;

    int sample_w;
    int sample_h;
    int sample_count;
} Sample;

typedef struct _Kernel
{
    double *W; //卷积矩阵
    double *dW; //delta W
} Kernel;

typedef struct _Map
{
    double *data;
    double *error;  // 节点误差项矩阵
    double  b; // bias
    double  db; //delta b
} Map;

typedef struct _Layer
{
    int map_w;
    int map_h;
    int map_count;
    Map *map;

    int kernel_w;
    int kernel_h;
    int kernel_count;
    Kernel *kernel;

    double *map_common;
} Layer;

const int batch_size = 10;//将训练数据集均分的份数
const int classes_count = 10;//输出的label种类有0,1,2,3... 十种
const int width  = 32;
const int height = 32;
const int train_sample_count = 60000;//训练集样本数
const int test_sample_count  = 10000;//测试集样本数

Layer input_layer, output_layer;
Layer c1_conv_layer, c3_conv_layer, c5_conv_layer;
Layer s2_pooling_layer, s4_pooling_layer;

//大端小端转换
int translateEndian_32(int i){
    return ((i & 0x000000FF) << 24 | (i & 0x0000FF00) << 8 | (i & 0x00FF0000) >> 8 | (i & 0xFF000000) >> 24);
}

void load_mnist_data(Sample * sample, const char * file_name){
    FILE *fp = NULL;
    fp = fopen(file_name, "rb");
    if(fp == NULL) {
        cout << "load mnist data failed." << endl;
        return;
    }
    int magic_number = 0;
    int sample_number = 0;
    int n_rows = 0, n_cols = 0;
    fread((int *)&magic_number, sizeof(magic_number), 1, fp);
    //文件存储格式为大端，Intel CPU架构存储为小端，所以需要将字节序反转
    magic_number = translateEndian_32(magic_number);
//    cout << "magic number = " << magic_number << endl;

    fread((int *)&sample_number, sizeof(sample_number), 1, fp);
    sample_number = translateEndian_32(sample_number);
//    cout << "sample number = " << sample_number << endl;

    fread((int *)&n_rows, sizeof(n_rows), 1, fp);
    n_rows = translateEndian_32(n_rows);
//    cout << "n_rows = " << n_rows << endl;

    fread((int *)&n_cols, sizeof(n_cols), 1, fp);
    n_cols = translateEndian_32(n_cols);
//    cout << "n_cols = " << n_cols << endl;

    int zero_padding = 2;
    int padded_matrix_size = (n_cols + 2 * zero_padding) * (n_rows + 2 * zero_padding);
    unsigned char temp;
    double normalize_max = 1, normalize_min = -1;
//    double normalize_max = 1.175, normalize_min = -0.1;

    //test
//    sample_number = 1;

    for(int k = 0; k < sample_number; k++){
        sample[k].data = (double *)malloc(padded_matrix_size * sizeof(double));
        memset(sample[k].data, 0, padded_matrix_size * sizeof(double));
        for(int i = 0; i < n_rows; i++){
            for(int j = 0; j < n_cols; j++){
                fread(&temp, 1, 1, fp);
//                cout << i << "  "<< j << "---" << (double)temp/255 << endl;
                sample[k].data[(i + zero_padding) * width + j + zero_padding] = (double)temp/255 * (normalize_max - normalize_min) + normalize_min;//把灰度归一化到[0, 1]
            }
        }
    }
    fclose(fp);
    fp = NULL;
}

void load_mnist_label(Sample * sample, const char * file_name){
    FILE *fp = NULL;
    fp = fopen(file_name, "rb");
    if(fp == NULL) {
        cout << "load mnist label failed." << endl;
        return;
    }
    int magic_number = 0;
    int sample_number = 0;
    fread((int *)&magic_number, sizeof(magic_number), 1, fp);
    //文件存储格式为大端，Intel CPU架构存储为小端，所以需要将字节序反转
    magic_number = translateEndian_32(magic_number);
//    cout << "magic number = " << magic_number << endl;

    fread((int *)&sample_number, sizeof(sample_number), 1, fp);
    sample_number = translateEndian_32(sample_number);
//    cout << "sample number = " << sample_number << endl;

    unsigned char temp;
    //test
//    sample_number = 100;
    for(int k = 0; k < sample_number; k++){
//        sample[k].label = (double *)malloc(classes_count * sizeof(double));
//        memset(sample[k].label, 0, classes_count * sizeof(double));
//        fread(&temp, 1, 1, fp);
//        sample[k].label[(int)temp] = 1;//todo 正样本label设成1，负样本label设成0
//        cout << "label == " << (int)temp << endl;

        sample[k].label = (double *)malloc(classes_count * sizeof(double));
        for (int i = 0; i < classes_count; i++) {
            sample[k].label[i] = -0.8;
        }

        fread((char*)&temp, sizeof(temp), 1, fp);
        sample[k].label[temp] = 0.8;
    }
    fclose(fp);
    fp = NULL;
}

//初始化卷积核，用较小的随机数填充卷积核的初始值
void init_kernel(double *kernel, int size, double weight_base) {
    for (int i = 0; i < size; i++) {
//        kernel[i] = (genrand_real1() - 0.5) * 2 * weight_base;
        //todo 产生高质量随机数方法
        kernel[i] = ((double)rand()/RAND_MAX - 0.5) * 2 * weight_base;
    }
}

//初始化Layer
//layer: 需要训练的层
//prevlayer_map_count: 上一层layer的map数
//map_count: 本层的map数
void init_layer(Layer *layer, int prevlayer_map_count, int map_count, int kernel_w, int kernel_h, int map_w, int map_h, bool is_pooling) {
    int mem_size = 0;

    const double scale = 6.0;
    int fan_in = 0;
    int fan_out = 0;
    if (is_pooling) {
        fan_in  = 4;
        fan_out = 1;
    }
    else {
        fan_in = prevlayer_map_count * kernel_w * kernel_h;
        fan_out = map_count * kernel_w * kernel_h;
    }
    int denominator = fan_in + fan_out;
    double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;

    layer->kernel_count = prevlayer_map_count * map_count;//需要训练的kernel数目
    layer->kernel_w = kernel_w;
    layer->kernel_h = kernel_h;
    layer->kernel = (Kernel *)malloc(layer->kernel_count * sizeof(Kernel));
    mem_size = layer->kernel_w * layer->kernel_h * sizeof(double);
    for (int i = 0; i < prevlayer_map_count; i++) {
        for (int j = 0; j < map_count; j++) {
            layer->kernel[i*map_count + j].W = (double *)malloc(mem_size);
            init_kernel(layer->kernel[i*map_count + j].W, layer->kernel_w*layer->kernel_h, weight_base);
            layer->kernel[i*map_count + j].dW = (double *)malloc(mem_size);
            memset(layer->kernel[i*map_count + j].dW, 0, mem_size);
        }
    }

    layer->map_count = map_count;
    layer->map_w = map_w;
    layer->map_h = map_h;
    layer->map = (Map *)malloc(layer->map_count * sizeof(Map));
    mem_size = layer->map_w * layer->map_h * sizeof(double);
    for (int i = 0; i < layer->map_count; i++) {
        layer->map[i].b = 0.0;
        layer->map[i].db = 0.0;
        layer->map[i].data = (double *)malloc(mem_size);
        layer->map[i].error = (double *)malloc(mem_size);
        memset(layer->map[i].data, 0, mem_size);
        memset(layer->map[i].error, 0, mem_size);
    }
    layer->map_common = (double *)malloc(mem_size);
    memset(layer->map_common, 0, mem_size);
}

void reset_params(Layer *layer)
{
    int mem_size = layer->kernel_w * layer->kernel_h * sizeof(double);
    for (int i = 0; i < layer->kernel_count; i++) {
        memset(layer->kernel[i].dW, 0, mem_size);
    }

    for (int i = 0; i < layer->map_count; i++) {
        layer->map[i].db = 0.0;
    }
}

void reset_weights() {
    reset_params(&c1_conv_layer);   // C1
    reset_params(&s2_pooling_layer);// S2
    reset_params(&c3_conv_layer);   // C3
    reset_params(&s4_pooling_layer);// S4
    reset_params(&c5_conv_layer);   // C5
    reset_params(&output_layer);    // Out
}

//激活函数
struct activation_func {
    /* scale: -0.8 ~ 0.8 和label初始值对应 */
    inline static double tan_h(double val) {
        double ep = exp(val);
        double em = exp(-val);

        return (ep - em) / (ep + em);
    }

    inline static double dtan_h(double val) {
        return 1.0 - val*val;
    }

    /* scale: 0.1 ~ 0.9 和label初始值对应 */
    inline static double relu(double val) {
        return val > 0.0 ? val : 0.0;
    }

    inline static double drelu(double val) {
        return val > 0.0 ? 1.0 : 0.0;
    }

    /* scale: 0.1 ~ 0.9 和label初始值对应 */
    inline double sigmoid(double val) {
        return 1.0 / (1.0 + exp(-val));
    }

    inline double dsigmoid(double val) {
        return val * (1.0 - val);
    }
};

//损失函数
struct loss_func {
    inline static double mse(double y, double t) {
        return (y - t) * (y - t) / 2;
    }

    inline static double dmse(double y, double t) {
        return y - t;
    }
};

//卷积函数， 四层for循环，，，，，，
void convn_valid(double *in_data, int in_w, int in_h, double *kernel, int kernel_w, int kernel_h, double *out_data, int out_w, int out_h) {
    double sum = 0.0;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            sum = 0.0;
            for (int n = 0; n < kernel_h; n++) {
                for (int m = 0; m < kernel_w; m++) {
                    sum += in_data[(i + n)*in_w + j + m] * kernel[n*kernel_w + m];
                }
            }
            out_data[i*out_w + j] += sum;
        }
    }
}

#define O true
#define X false
//S2到C3不完全连接，经验映射表
bool connection_table[6*16] = {
                O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
                O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
                O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
                X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
                X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
                X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

/////////////////////
////
////forward progation
////
/////////////////////
//卷积层正向传播函数
void conv_fprop(Layer *prev_layer, Layer *layer, bool *pconnection) {
    int index = 0;
    int size = layer->map_w * layer->map_h;
    for (int i = 0; i < layer->map_count; i++) {
        memset(layer->map_common, 0, size*sizeof(double));
        for (int j = 0; j < prev_layer->map_count; j++) {
            index = j*layer->map_count + i;
            if (pconnection != NULL && !pconnection[index]) {
                continue;
            }
            convn_valid(
                    prev_layer->map[j].data, prev_layer->map_w, prev_layer->map_h,
                    layer->kernel[index].W, layer->kernel_w, layer->kernel_h,
                    layer->map_common, layer->map_w, layer->map_h);
        }

        for (int k = 0; k < size; k++) {
            layer->map[i].data[k] = activation_func::tan_h(layer->map_common[k] + layer->map[i].b);//采用双曲正切函数来表示激活函数
        }
    }
}

//pooling层前向传播 取最大值
void max_pooling_fprop(Layer *prev_layer, Layer *layer) {
    int map_w = layer->map_w;
    int map_h = layer->map_h;
    int upmap_w = prev_layer->map_w;

    for (int k = 0; k < layer->map_count; k++) {
        for (int i = 0; i < map_h; i++) {
            for (int j = 0; j < map_w; j++) {
                double max_value = prev_layer->map[k].data[2*i*upmap_w + 2*j];
                for (int n = 2*i; n < 2*(i + 1); n++) {
                    for (int m = 2*j; m < 2*(j + 1); m++) {
                        max_value = MAX(max_value, prev_layer->map[k].data[n*upmap_w + m]);
                    }
                }
                layer->map[k].data[i*map_w + j] = activation_func::tan_h(max_value);
            }
        }
    }
}

//全连接层前向传播
void fully_connected_fprop(Layer *prev_layer, Layer *layer) {
    for (int i = 0; i < layer->map_count; i++) {
        double sum = 0.0;
        for (int j = 0; j < prev_layer->map_count; j++) {
            sum += prev_layer->map[j].data[0] * layer->kernel[j*layer->map_count + i].W[0];
        }
        sum += layer->map[i].b;
        layer->map[i].data[0] = activation_func::tan_h(sum);
    }
}

void forward_propagation(){
    // In-->C1
    conv_fprop(&input_layer, &c1_conv_layer, NULL);

    // C1-->S2
    max_pooling_fprop(&c1_conv_layer, &s2_pooling_layer);

    // S2-->C3
    conv_fprop(&s2_pooling_layer, &c3_conv_layer, connection_table);

    // C3-->S4
    max_pooling_fprop(&c3_conv_layer, &s4_pooling_layer);

    // S4-->C5
    conv_fprop(&s4_pooling_layer, &c5_conv_layer, NULL);

    //中间省略一层全连接层

    // C5-->Out
    fully_connected_fprop(&c5_conv_layer, &output_layer);
}




//////////////////
////
////back progation
////
//////////////////

void fully_connected_bprop(Layer *layer, Layer *prev_layer) {
    // 更新delta
    for (int i = 0; i < prev_layer->map_count; i++) {
        prev_layer->map[i].error[0] = 0.0;
        for (int j = 0; j < layer->map_count; j++) {
            prev_layer->map[i].error[0] += layer->map[j].error[0] * layer->kernel[i*layer->map_count + j].W[0];//全连接隐藏层节点误差项
        }
        prev_layer->map[i].error[0] *= activation_func::dtan_h(prev_layer->map[i].data[0]);//全连接隐藏层节点误差项
    }

    // 更新delta W
    for (int i = 0; i < prev_layer->map_count; i++) {
        for (int j = 0; j < layer->map_count; j++) {
            layer->kernel[i*layer->map_count + j].dW[0] += layer->map[j].error[0] * prev_layer->map[i].data[0];
        }
    }

    // 更新delta bias
    for (int i = 0; i < layer->map_count; i++) {
        layer->map[i].db += layer->map[i].error[0];
    }
}

void conv_bprop(Layer *layer, Layer *prev_layer, bool *pconnection) {
    int index = 0;
    int size = prev_layer->map_w * prev_layer->map_h;

    // delta
    for (int i = 0; i < prev_layer->map_count; i++) {
        memset(prev_layer->map_common, 0, size*sizeof(double));
        for (int j = 0; j < layer->map_count; j++) {
            index = i*layer->map_count + j;
            if (pconnection != NULL && !pconnection[index]) {
                continue;
            }

            for (int n = 0; n < layer->map_h; n++) {
                for (int m = 0; m < layer->map_w; m++) {
                    double error = layer->map[j].error[n*layer->map_w + m];
                    for (int ky = 0; ky < layer->kernel_h; ky++) {
                        for (int kx = 0; kx < layer->kernel_w; kx++) {
                            prev_layer->map_common[(n + ky)*prev_layer->map_w + m + kx] += error * layer->kernel[index].W[ky*layer->kernel_w + kx];
                        }
                    }
                }
            }
        }

        for (int k = 0; k < size; k++) {
            prev_layer->map[i].error[k] = prev_layer->map_common[k] * activation_func::dtan_h(prev_layer->map[i].data[k]);
        }
    }

    // dW
    for (int i = 0; i < prev_layer->map_count; i++) {
        for (int j = 0; j < layer->map_count; j++) {
            index = i*layer->map_count + j;
            if (pconnection != NULL && !pconnection[index]) {
                continue;
            }
            convn_valid(
                    prev_layer->map[i].data, prev_layer->map_w, prev_layer->map_h,
                    layer->map[j].error, layer->map_w, layer->map_h,
                    layer->kernel[index].dW, layer->kernel_w, layer->kernel_h);
        }
    }

    // db
    size = layer->map_w * layer->map_h;
    for (int i = 0; i < layer->map_count; i++) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
            sum += layer->map[i].error[k];
        }
        layer->map[i].db += sum;
    }
}


void max_pooling_bprop(Layer *layer, Layer *prev_layer) {
    int map_w = layer->map_w;
    int map_h = layer->map_h;
    int upmap_w = prev_layer->map_w;

    for (int k = 0; k < layer->map_count; k++) {
        // delta
        for (int i = 0; i < map_h; i++) {
            for (int j = 0; j < map_w; j++) {
                int row = 2*i, col = 2*j;
                double max_value = prev_layer->map[k].data[row*upmap_w + col];
                for (int n = 2*i; n < 2*(i + 1); n++) {
                    for (int m = 2*j; m < 2*(j + 1); m++) {
                        if (prev_layer->map[k].data[n*upmap_w + m] > max_value) {
                            row = n;
                            col = m;
                            max_value = prev_layer->map[k].data[n*upmap_w + m];
                        }
                        else {
                            prev_layer->map[k].error[n*upmap_w + m] = 0.0;
                        }
                    }
                }
                prev_layer->map[k].error[row*upmap_w + col] = layer->map[k].error[i*map_w + j] * activation_func::dtan_h(max_value);//todo 这里需要乘以tan_h的导数函数吗
            }
        }
    }

    //Pooling层只需要将误差项传递到上一层，没有梯度的计算
}

void backward_propagation(double *label) {
    for (int i = 0; i < output_layer.map_count; i++) {
//        cout << "bi == " << i << endl;
//        cout << "output_layer.map[i].data[0] = " << output_layer.map[i].data[0] << endl;
//        cout << "label[i] = " << label[i] << endl;
        output_layer.map[i].error[0] = loss_func::dmse(output_layer.map[i].data[0], label[i]) * activation_func::dtan_h(output_layer.map[i].data[0]);
//        cout << "output_layer.map[i].error[0]= " << output_layer.map[i].error[0] << endl;
    }

    // Out-->C5
    fully_connected_bprop(&output_layer, &c5_conv_layer);

    // C5-->S4
    conv_bprop(&c5_conv_layer, &s4_pooling_layer, NULL);

    // S4-->C3
    max_pooling_bprop(&s4_pooling_layer, &c3_conv_layer);

    // C3-->S2
    conv_bprop(&c3_conv_layer, &s2_pooling_layer, connection_table);

    // S2-->C1
    max_pooling_bprop(&s2_pooling_layer, &c1_conv_layer);

    // C1-->In
    conv_bprop(&c1_conv_layer, &input_layer, NULL);
}



inline double gradient_descent(double W, double dW, double alpha, double lambda) {
    return W - alpha * (dW + lambda * W);
}

void update_params(Layer *layer, double learning_rate) {
    const double lambda = 0.0;//正则项，暂时设为零

    // W
    int size = layer->kernel_w * layer->kernel_h;
    for (int i = 0; i < layer->kernel_count; i++) {
        for (int k = 0; k < size; k++) {
            layer->kernel[i].W[k] = gradient_descent(layer->kernel[i].W[k], layer->kernel[i].dW[k] / batch_size, learning_rate, lambda);
        }
    }

    // b
    for (int i = 0; i < layer->map_count; i++) {
        layer->map[i].b = gradient_descent(layer->map[i].b, layer->map[i].db / batch_size, learning_rate, lambda);
    }
}

void update_weights(double learning_rate) {
    update_params(&c1_conv_layer, learning_rate);   // C1
    update_params(&s2_pooling_layer, learning_rate);// S2
    update_params(&c3_conv_layer, learning_rate);   // C3
    update_params(&s4_pooling_layer, learning_rate);// S4
    update_params(&c5_conv_layer, learning_rate);   // C5
    update_params(&output_layer, learning_rate);    // Out
}


void train(Sample *train_sample, double learning_rate) {
    // 随机打乱样本顺序
    int i = 0, j = 0, temp = 0;
    int *rand_perm = (int *)malloc(train_sample->sample_count * sizeof(int));
    for (i = 0; i < train_sample->sample_count; i++) {
        rand_perm[i] = i;
    }

    for (i = 0; i < train_sample->sample_count; i++) {
        //todo 这个随机交换位置的函数机制公平性还有待考虑
        j = rand() % (train_sample->sample_count - i) + i;
        temp = rand_perm[j];
        rand_perm[j] = rand_perm[i];
        rand_perm[i] = temp;
    }

    // 迭代训练
    int batch_count = train_sample->sample_count / batch_size;
    int data_mem_size = train_sample->sample_w * train_sample->sample_h * sizeof(double);
    for (i = 0; i < batch_count; i++) {
        // 重置参数dW和db
        reset_weights();
//        cout << "i == " << i << endl;
        for (j = 0; j < batch_size; j++) {
            // 填充数据
            int index = i*batch_size + j;
            memcpy(input_layer.map[0].data, train_sample[rand_perm[index]].data, data_mem_size);

            // 前向/反向传播计算
            forward_propagation();
            backward_propagation(train_sample[rand_perm[index]].label);
        }

        // 更新权值
        update_weights(learning_rate);

        if (i % 1000 == 0) {
            printf("progress...%d/%d \n", i, batch_count);
        }
    }

    free(rand_perm);
    rand_perm = NULL;
}

int find_index(Layer * layer){
    int index = 0;
    double max_val = *(layer->map[0].data);
    for (int i = 1; i < layer->map_count; i++) {
        if (*(layer->map[i].data) > max_val) {
            max_val = *(layer->map[i].data);
            index = i;
        }
    }

    return index;
}

int find_index(double *label) {
    int index = 0;
    double max_val = label[0];
    for (int i = 1; i < classes_count; i++) {
        if (label[i] > max_val) {
            max_val = label[i];
            index = i;
        }
    }

    return index;
}

void predict(Sample * test_sample) {
    int num_success = 0, predict = 0, actual = 0;
    int data_mem_size = test_sample->sample_h * test_sample->sample_w * sizeof(double);
    int *confusion_matrix = (int *)malloc(classes_count * classes_count * sizeof(int));
    memset(confusion_matrix, 0, classes_count * classes_count * sizeof(int));

    for (int i = 0; i < test_sample->sample_count; i++) {
        memcpy(input_layer.map[0].data, test_sample[i].data, data_mem_size);
        forward_propagation();

        predict = find_index(&output_layer);
        actual = find_index(test_sample[i].label);
        if (predict == actual) {
            num_success++;
        }

        confusion_matrix[predict*classes_count + actual]++;
    }

    printf("accuracy: %d/%d\n", num_success, test_sample->sample_count);
    printf("\n   *  ");
    for (int i = 0; i < classes_count; i++)
    {
        printf("%4d  ", i);
    }

    printf("\n");
    for (int i = 0; i < classes_count; i++)
    {
        printf("%4d  ", i);
        for (int j = 0; j < classes_count; j++)
        {
            printf("%4d  ", confusion_matrix[i*classes_count + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(confusion_matrix);
    confusion_matrix = NULL;
}

void release_layer(Layer *layer)
{
    for (int i = 0; i < layer->kernel_count; i++)
    {
        free(layer->kernel[i].W);
        free(layer->kernel[i].dW);
        layer->kernel[i].W = NULL;
        layer->kernel[i].dW = NULL;
    }
    free(layer->kernel);
    layer->kernel = NULL;

    for (int i = 0; i < layer->map_count; i++)
    {
        free(layer->map[i].data);
        free(layer->map[i].error);
        layer->map[i].data = NULL;
        layer->map[i].error = NULL;
    }
    free(layer->map_common);
    layer->map_common = NULL;
    free(layer->map);
    layer->map = NULL;
}

//int checkCPU() {
//    unsigned int i = 1;
//    if(*((char*)&i) == 0)
//    {
//        printf("this is big endian. \n");
//    }
//    else if(*((char*)&i) == 1)
//    {
//        printf(" this is little endian. \n");
//    }
//    else
//    {
//        printf("sorry, i do not know . \n");
//    }
//    return 0;
//}


int main() {
    int kernel_w = 0, kernel_h = 0;
    double learning_rate = 0.01 * sqrt((double) batch_size);

    //加载训练数据集文件
    Sample * train_sample = (Sample *)malloc(train_sample_count * sizeof(Sample));
    memset(train_sample, 0, train_sample_count * sizeof(Sample));
    train_sample->sample_h = height;
    train_sample->sample_w = width;
    train_sample->sample_count = train_sample_count;

    const char* train_sample_path = "/home/shawn/CNN_data/train-images.idx3-ubyte";
    const char* train_label_path = "/home/shawn/CNN_data/train-labels.idx1-ubyte";

    //    checkCPU();
    load_mnist_data(train_sample, train_sample_path);
    load_mnist_label(train_sample, train_label_path);


    //加载测试数据集文件
    Sample * test_sample = (Sample *)malloc(test_sample_count * sizeof(Sample));
    memset(test_sample, 0, test_sample_count * sizeof(Sample));
    test_sample->sample_h = height;
    test_sample->sample_w = width;
    test_sample->sample_count = test_sample_count;

    const char* test_sample_path = "/home/shawn/CNN_data/t10k-images.idx3-ubyte";
    const char* test_label_path = "/home/shawn/CNN_data/t10k-labels.idx1-ubyte";

    load_mnist_data(test_sample, test_sample_path);
    load_mnist_label(test_sample, test_label_path);

    // 输入层In
    kernel_w = 0;
    kernel_h = 0;
    init_layer(&input_layer, 0, 1, kernel_w, kernel_h, width, height, false);

    // 卷积层C1
    kernel_w = 5;
    kernel_h = 5;
    init_layer(&c1_conv_layer, 1, 6, kernel_w, kernel_h, input_layer.map_w - kernel_w + 1, input_layer.map_h - kernel_h + 1, false);

    // 采样层S2
    kernel_w = 1;
    kernel_h = 1;
    init_layer(&s2_pooling_layer, 1, 6, kernel_w, kernel_h, c1_conv_layer.map_w / 2, c1_conv_layer.map_h / 2, true);

    // 卷积层C3
    kernel_w = 5;
    kernel_h = 5;
    init_layer(&c3_conv_layer, 6, 16, kernel_w, kernel_h, s2_pooling_layer.map_w - kernel_w + 1, s2_pooling_layer.map_h - kernel_h + 1, false);

    // 采样层S4
    kernel_w = 1;
    kernel_h = 1;
    init_layer(&s4_pooling_layer, 1, 16, kernel_w, kernel_h, c3_conv_layer.map_w / 2, c3_conv_layer.map_h / 2, true);

    // 全连接层C5
    kernel_w = 5;
    kernel_h = 5;
    init_layer(&c5_conv_layer, 16, 120, kernel_w, kernel_h, s4_pooling_layer.map_w - kernel_w + 1, s4_pooling_layer.map_h - kernel_h + 1, false);

    // 输出层Out
    kernel_w = 1;
    kernel_h = 1;
    init_layer(&output_layer, 120, 10, kernel_w, kernel_h, 1, 1, false);

    // 训练及测试
    clock_t start_time = 0;
    const int epoch = 20;
    for (int i = 0; i < epoch; i++) {
        printf("train epoch is %d ************************************************\n", i + 1);
        start_time = clock();
        train(train_sample, learning_rate);
        printf("train time is....%f s\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

        printf("predict epoch is %d ************************************************\n", i + 1);
        start_time = clock();
        predict(test_sample);
        printf("predict time is....%f s\n\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

        learning_rate *= 0.85;//逐步缩小学习率
    }

    for (int i = 0; i < train_sample_count; i++) {
        free(train_sample[i].data);
        free(train_sample[i].label);
        train_sample[i].data = NULL;
        train_sample[i].label = NULL;
    }
    free(train_sample);

    for (int i = 0; i < test_sample_count; i++) {
        free(test_sample[i].data);
        free(test_sample[i].label);
        test_sample[i].data = NULL;
        test_sample[i].label = NULL;
    }
    free(test_sample);

    release_layer(&input_layer);
    release_layer(&c1_conv_layer);
    release_layer(&c3_conv_layer);
    release_layer(&c5_conv_layer);
    release_layer(&s2_pooling_layer);
    release_layer(&s4_pooling_layer);
    release_layer(&output_layer);

//    system("pause");

    return 0;
}