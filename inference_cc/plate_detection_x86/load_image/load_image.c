#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 3


#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

int main() {

    int width, height, channels;
    uint8_t* rgb_image = stbi_load("BAG-7751.jpg", &width, &height, &channels, 3);

    if(rgb_image == NULL){
        printf("Imagem nao carregada\n");
        exit(1);
    }

    printf("Imagem carregada com (%d, %d, %d)\n", width, height, channels);
    
    //   stbir_resize_uint8(input_pixels, in_w, in_h, 0, output_pixels, out_w, out_h, 0, num_channels)
    int out_w = 320;
    int out_h = 320;
    unsigned char* rgb_image_resize = (unsigned char *) malloc(out_w * out_h * CHANNEL_NUM);

    stbir_resize_uint8(rgb_image, width , height , width*CHANNEL_NUM, rgb_image_resize, out_w, out_h, out_w*CHANNEL_NUM, CHANNEL_NUM);
    stbir_resize_uint8(rgb_image, width , height , 0, rgb_image_resize, out_w, out_h, 0, CHANNEL_NUM);
    
    stbi_write_png("BAG-7751_gravou.jpg", width, height, CHANNEL_NUM, rgb_image, width*CHANNEL_NUM);
    stbi_write_png("BAG-7751_gravou_resize.jpg", out_w, out_h, CHANNEL_NUM, rgb_image_resize, out_w*CHANNEL_NUM);

    printf("Imagem com resize gravada");


    stbi_image_free(rgb_image);
    stbi_image_free(rgb_image_resize);
    return 0;
}