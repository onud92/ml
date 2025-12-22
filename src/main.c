/*
 * Copyright (c) 2016 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/console/console.h>

#ifdef CONFIG_ML_HELLO
    #include "ml_hello.h"
#endif

/* Sleep time in ms between led toggle */
#define BLINK_SLEEP_TIME_MS   200

/* The devicetree node identifier for the "led0" alias. */
#define LED0_NODE DT_ALIAS(led0)

/* Stack size for the blinking thread */
#define BLINK_THREAD_STACK_SIZE 512

/* Define stack areas for the threads at compile time */
K_THREAD_STACK_DEFINE(blink_stack, BLINK_THREAD_STACK_SIZE);

/* Declare thread data structs */
static struct k_thread blink_thread;

/* Declare thread function */
void blink_thread_start(void *arg_1, void *arg_2, void *arg_3);


/*
 * A build error on this line means your board is unsupported.
 * See the sample documentation for information on how to fix this.
 */
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

int main(void)
{
	int ret;
	k_tid_t blink_tid;

	/* make sure that the GPIO was initialized */
	if (!gpio_is_ready_dt(&led))
	{
		printf("Error: Init led failed.\r\n");
		return 0;
	}
	/* set the GPIO as output */
	ret = gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
	if (ret < 0) 
	{
		printf("Error: Configure led failed.\r\n");
		return 0;
	}

#ifdef CONFIG_ML_HELLO
	/* Hello message for my ml application */
	ml_hello();
#endif

	/* Start the blink thread */
	blink_tid = k_thread_create(
					&blink_thread, 				/* thread struct */
					blink_stack,				/* stack */
					K_THREAD_STACK_SIZEOF(blink_stack), /* size of the stack */
					blink_thread_start,			/* entry point */
					NULL,						/* arg1 */
					NULL,						/* arg2 */
					NULL,						/* arg3 */
					7,							/* priority */
					0,							/* thread options */
					K_NO_WAIT					/* delay */
				);

	while (1) 
	{
		printf("Hello from main \r\n");
		k_msleep(1000);
	}
	return 0;
}



void blink_thread_start(void *arg_1, void *arg_2, void *arg_3)
{
	int ret;

	while (1) {
		ret = gpio_pin_toggle_dt(&led);
		if (ret < 0) {
			printf("Error: Toggle led failed.\r\n");
		}

		k_msleep(BLINK_SLEEP_TIME_MS);
	}
}