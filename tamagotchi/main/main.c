
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_system.h"
#include <string.h>
#include "esp_log.h"
#include <time.h>


//OLED
//Include for graphics https://github.com/nopnop2002
#include "ssd1306.h"
#include "font8x8_basic.h"
#include "dino_sprites.c"
SSD1306_t dev;


// JOYSTICK
// Define the ADC channels for the joystick axes
#define ADC_CHANNEL_X ADC2_CHANNEL_6  // GPIO14
#define ADC_CHANNEL_Y ADC2_CHANNEL_3  // GPIO15
#define JOYSTICK_BUTTON_PIN 13
// ADC width and attenuation for joystick
#define ADC_WIDTH ADC_WIDTH_BIT_12    // 12-bit resolution
#define ADC_ATTEN ADC_ATTEN_DB_11     // 11dB attenuation (0-3.9V input range)

/*additional config

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
int selected_button = 0;


//DINO VARIABLES
int dino_x = 10;
int dino_y = 24;
int dino_direction = 0; // 0: right, 1: up, 2: left, 3: down
unsigned char *bitmap_right_tail_down;


// Function to flip the bitmap along the middle vertical axis
unsigned char* flip_bitmap_vertical(const unsigned char* input, int width, int height) {
    int bytesPerRow = width / 8;  // Number of bytes per row, since 8 pixels per byte
     unsigned char* output = (unsigned char*)malloc(bytesPerRow*height);
    if (output == NULL) {
        return NULL;  // Handle allocation failure
    }
    // Flip along the Y-axis (vertical flip)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < bytesPerRow; x++) {
            // Reverse the order of rows
            output[y * bytesPerRow + x] = input[(height - 1 - y) * bytesPerRow + x];
        }
    }
    return output;
}

unsigned char* flip_bitmap_horizontal(unsigned char* input, int width, int height) {
    int bytesPerRow = width / 8;
    unsigned char* output = (unsigned char*)malloc(bytesPerRow*height);
    if (output == NULL) {
        return NULL;  // Handle allocation failure
    }
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
    return output;
}

void initialize_bitmaps() {
    // Assuming bitmap_left_tail_down is already defined
    bitmap_right_tail_down = flip_bitmap_horizontal(bitmap_left_tail_down, 96, 44);
    
    if (bitmap_right_tail_down == NULL) {
        // Handle allocation failure
        ESP_LOGE(tag, "Failed to allocate memory for bitmaps");
        // Proper error handling (e.g., restart the device)
    }
}

void draw_box(SSD1306_t *dev, int x, int y, int w, int h) {
    for (int i = 0; i < w; i++) {
        _ssd1306_pixel(dev, x + i, y, false);
        _ssd1306_pixel(dev, x + i, y + h, false);
    }
    for (int i = 0; i < h; i++) {
        _ssd1306_pixel(dev, x, y + i, false);
        _ssd1306_pixel(dev, x + w, y + i, false);
    }
}

void draw_menu_buttons() {
    // Draw four boxes on the top part of the screen
    for (int i = 0; i < 4; i++) {
        int x = (i * 31) + 4;
        int y = -1;
        draw_box(&dev, x, y, 30, 12);
    }
    
    // Add labels to the buttons
    ssd1306_display_text(&dev, 0, 12, ">>", 2, false);
    ssd1306_display_text(&dev, 0, 44, "-<", 2, false);
    ssd1306_display_text(&dev, 0, 75, "zz", 2, false);
    ssd1306_display_text(&dev, 0, 105, "^^", 2, false);
}

void move_dino(){
    int last_dino_direction = -1;
    // Randomly change direction occasionally
    if (rand() % 10 == 0) {
        dino_direction = rand() % 4;
    }

    // Move based on current direction
    switch (dino_direction) {
        case 0: // Right
            dino_x += 1;
            if (dino_x > 30) dino_x = 30;
            last_dino_direction = 0;
            break;
        case 1: // Up
            dino_y -= 1;
            last_dino_direction = last_dino_direction;
            if (dino_y < 12) dino_y = 12; // Don't go into menu area
            break;
        case 2: // Left
            dino_x -= 1;
            if (dino_x < 0) dino_x = 0;
            last_dino_direction = 2;
            break;
        case 3: // Down
            dino_y += 1;
            last_dino_direction = last_dino_direction;
            if (dino_y > 20) dino_y = 20;
            break;
    }

    // Draw new position
    unsigned char* dino_bitmap;
    if (dino_direction == 0) {
        ESP_LOGE(tag, "right");
        dino_bitmap = bitmap_right_tail_down;
    }else if (dino_direction == 2){
        ESP_LOGE(tag, "left");
        dino_bitmap = bitmap_left_tail_down;
    }else if((dino_direction == 1 || dino_direction == 3) && (last_dino_direction == 0 || last_dino_direction == -1)){
        ESP_LOGE(tag, "u/d right");
        dino_bitmap = bitmap_right_tail_down;
    }else if((dino_direction == 1 || dino_direction == 3) && (last_dino_direction == 2 || last_dino_direction == -1)){
        ESP_LOGE(tag, "u/d left");
        dino_bitmap = bitmap_left_tail_down;
    }else{
        ESP_LOGE(tag, "none");
        dino_bitmap = bitmap_left_tail_down;
    }
    ssd1306_bitmaps(&dev, dino_x, dino_y, dino_bitmap, 96, 44, false);
}

void handle_joystick() {
    int x_value, y_value;
    adc2_get_raw(ADC_CHANNEL_X, ADC_WIDTH_BIT_12, &x_value);
    adc2_get_raw(ADC_CHANNEL_Y, ADC_WIDTH_BIT_12, &y_value);
    
    // Highlight the selected button
    for (int i = 0; i < 4; i++) {
        int x = (i * 31) + 8;
        int y = 8;
        _ssd1306_line(&dev, x, y, x+21, y, true);
        if(i == selected_button){
            _ssd1306_line(&dev, x, y, x+21, y, false);
        }
    }
    
    // Update selection based on joystick input
    if (x_value < 1500) {
        selected_button--;
        if(selected_button < 0) selected_button = 3;
    } else if (x_value > 2000) {
        selected_button++;
        if(selected_button > 3) selected_button = 0;
    }

    // Check if the joystick button is pressed
    if (gpio_get_level(JOYSTICK_BUTTON_PIN) == 0) {  // Assuming active low
        printf("Button pressed! Selected menu item: %d\n", selected_button);
        vTaskDelay(pdMS_TO_TICKS(10));  // Debounce delay
    }

    //printf("selected button: %d\n", selected_button);
    ssd1306_show_buffer(&dev);
}

void app_main() {
    // Configure ADC for the X and Y axis channels
    adc2_config_channel_atten(ADC_CHANNEL_X, ADC_ATTEN);
    adc2_config_channel_atten(ADC_CHANNEL_Y, ADC_ATTEN);
    // Configure GPIO for joystick button
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_DISABLE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << JOYSTICK_BUTTON_PIN),
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    gpio_config(&io_conf);
    srand(time(NULL)); // Initialize random seed
    //config for i2c
    ESP_LOGI(tag, "INTERFACE is i2c");
	ESP_LOGI(tag, "CONFIG_SDA_GPIO=%d",CONFIG_SDA_GPIO);
	ESP_LOGI(tag, "CONFIG_SCL_GPIO=%d",CONFIG_SCL_GPIO);
	ESP_LOGI(tag, "CONFIG_RESET_GPIO=%d",CONFIG_RESET_GPIO);

    //config oled for 128x64
	i2c_master_init(&dev, CONFIG_SDA_GPIO, CONFIG_SCL_GPIO, CONFIG_RESET_GPIO);
    ESP_LOGI(tag, "Panel is 128x64");
	ssd1306_init(&dev, 128, 64);

    // Clear the screen
    ssd1306_clear_screen(&dev, false);    
    draw_menu_buttons();
    
    initialize_bitmaps();


    while (1) {
        move_dino();
        handle_joystick();
    }


    //for(int i=0;i<width;i++) {
    //    ssd1306_wrap_arround(&dev, SCROLL_LEFT, 2, 7, 1);
    //}
}
