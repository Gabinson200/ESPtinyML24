
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_system.h"
#include <string.h>
#include "esp_log.h"

//Include for graphics https://github.com/nopnop2002
#include "ssd1306.h"
#include "font8x8_basic.h"
#include "dino_sprites.c"
// Define the ADC channels for the joystick axes
#define ADC_CHANNEL_X ADC2_CHANNEL_6  // GPIO14
#define ADC_CHANNEL_Y ADC2_CHANNEL_3  // GPIO15

// ADC width and attenuation for joystick
#define ADC_WIDTH ADC_WIDTH_BIT_12    // 12-bit resolution
#define ADC_ATTEN ADC_ATTEN_DB_11     // 11dB attenuation (0-3.9V input range)


// Function to flip the bitmap along the middle vertical axis
void flipBitmapVertical(const unsigned char* input, unsigned char* output, int width, int height) {
    int bytesPerRow = width / 8;  // Number of bytes per row, since 8 pixels per byte
    // Flip along the Y-axis (vertical flip)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < bytesPerRow; x++) {
            // Reverse the order of rows
            output[y * bytesPerRow + x] = input[(height - 1 - y) * bytesPerRow + x];
        }
    }
}


void flipBitmapHorizontalTwo(unsigned char* input, unsigned char* output, int width, int height) {
    int bytesPerRow = width / 8;
    
    // Iterate over each row in the bitmap
    for (int row = 0; row < height; row++) {
        // Calculate the start index of the current row
        int rowStart = row * bytesPerRow;

        // Flip the elements horizontally in each row
        for (int i = 0; i < bytesPerRow / 2; i++) {
            // Swap elements at position 'i' and 'bytesPerRow - 1 - i' within the row
            unsigned char temp = input[rowStart + i];
            output[rowStart + i] = ssd1306_rotate_byte(input[rowStart + bytesPerRow - 1 - i]);
            output[rowStart + bytesPerRow - 1 - i] = ssd1306_rotate_byte(temp);
        }

        // If there's an odd number of bytes per row, process the middle element
        if (bytesPerRow % 2 != 0) {
            output[rowStart + bytesPerRow / 2] = ssd1306_rotate_byte(input[rowStart + bytesPerRow / 2]);
        }
    }
}

/*
 You have to set this config value with menuconfig for oled
 CONFIG_INTERFACE

 for i2c
 CONFIG_MODEL
 CONFIG_SDA_GPIO
 CONFIG_SCL_GPIO
 CONFIG_RESET_GPIO

 for SPI
 CONFIG_CS_GPIO
 CONFIG_DC_GPIO
 CONFIG_RESET_GPIO
*/

#define CONFIG_SDA_GPIO 33
#define CONFIG_SCL_GPIO 32
#define CONFIG_RESET_GPIO -1

#define tag "SSD1306"


void draw_box(SSD1306_t *dev, int x, int y, int w, int h) {
    for (int i = 0; i < w; i++) {
        _ssd1306_pixel(dev, x + i, y, false);
        _ssd1306_pixel(dev, x + i, y + h - 1, false);
    }
    for (int i = 0; i < h; i++) {
        _ssd1306_pixel(dev, x, y + i, false);
        _ssd1306_pixel(dev, x + w - 1, y + i, false);
    }
}



void app_main() {
    // Configure ADC for the X and Y axis channels
    adc2_config_channel_atten(ADC_CHANNEL_X, ADC_ATTEN);
    adc2_config_channel_atten(ADC_CHANNEL_Y, ADC_ATTEN);

    SSD1306_t dev;
	int center, top, bottom;
	char lineChar[20];
    //config for i2c
    ESP_LOGI(tag, "INTERFACE is i2c");
	ESP_LOGI(tag, "CONFIG_SDA_GPIO=%d",CONFIG_SDA_GPIO);
	ESP_LOGI(tag, "CONFIG_SCL_GPIO=%d",CONFIG_SCL_GPIO);
	ESP_LOGI(tag, "CONFIG_RESET_GPIO=%d",CONFIG_RESET_GPIO);
	i2c_master_init(&dev, CONFIG_SDA_GPIO, CONFIG_SCL_GPIO, CONFIG_RESET_GPIO);
    //config oled for 128x64
    ESP_LOGI(tag, "Panel is 128x64");
	ssd1306_init(&dev, 128, 64);
    top = 2;
	center = 3;
	bottom = 8;
	 // Box dimensions
    int box_width = 31;
    int box_height = 14;
    unsigned char current_dino[688];  // 128x43 means (128 * 43) / 8 = 688 bytes
    // Clear the screen
    ssd1306_clear_screen(&dev, false);    

    // Draw four boxes on the top part of the screen
    for (int i = 0; i < 4; i++) {
        int x = (i * box_width) + 2;
        int y = -4;
        draw_box(&dev, x, y, box_width, box_height);
    }

    ssd1306_display_text(&dev, 0, 10, ">>", 2, false);
    ssd1306_display_text(&dev, 0, 42, "-<", 2, false);
    ssd1306_display_text(&dev, 0, 73, "zz", 2, false);
    ssd1306_display_text(&dev, 0, 103, "^^", 2, false);


    // Show the buffer on the display
    ssd1306_show_buffer(&dev);

    int width = ssd1306_get_width(&dev);
    
    // Call the flip function
    //flipBitmap(bitmap_left_tail_up, bitmap_right_tail_up, 128, 43, FLIP_HORIZONTAL);
    flipBitmapHorizontalTwo(bitmap_left_tail_up, current_dino, 128, 43);
    ssd1306_bitmaps(&dev, 0, 20, current_dino, 128, 43, false);


    for (int i = 0; i < 64; i++) {
        _ssd1306_pixel(&dev, i, 48, false);
    }
    for (int i = 120; i < width; i++) {
        _ssd1306_pixel(&dev, i, 48, false);
    }

    for(int i=0;i<width;i++) {
        ssd1306_wrap_arround(&dev, SCROLL_LEFT, 2, 7, 1);
    }

    vTaskDelay(3000 / portTICK_PERIOD_MS);

    flipBitmapVertical(bitmap_left_tail_up, current_dino, 128, 43);
    ssd1306_bitmaps(&dev, 0, 20, current_dino, 128, 43, false);

    for (int i = 0; i < 64; i++) {
        _ssd1306_pixel(&dev, i, 48, false);
    }
    for (int i = 120; i < width; i++) {
        _ssd1306_pixel(&dev, i, 48, false);
    }

    for(int i=0;i<width;i++) {
        ssd1306_wrap_arround(&dev, SCROLL_LEFT, 2, 7, 1);
    }

   
    vTaskDelay(2000 / portTICK_PERIOD_MS);
	// Invert
	//ssd1306_clear_screen(&dev, true);
	//ssd1306_contrast(&dev, 0xff);
	//ssd1306_display_text(&dev, center, "  Good Bye!!", 12, true);
	//vTaskDelay(5000 / portTICK_PERIOD_MS);
    /*
    while (1) {
        // Read the ADC values for X and Y axes
        int joystick_x, joystick_y;

        adc2_get_raw(ADC_CHANNEL_X, ADC_WIDTH, &joystick_x);
        adc2_get_raw(ADC_CHANNEL_Y, ADC_WIDTH, &joystick_y);

        // Print the values to the serial monitor
        printf("Joystick X: %d, Joystick Y: %d\n", joystick_x, joystick_y);

        // Delay to avoid spamming the serial monitor
        vTaskDelay(pdMS_TO_TICKS(500));
    }
    */
}
