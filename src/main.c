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

/* Max. sleep time in ms between led toggle */
#define BLINK_MAX_SLEEP_TIME_MS   2000
/* Min. sleep time in ms between led toggle */
#define BLINK_MIN_SLEEP_TIME_MS   0

/* The devicetree node identifier for the "led0" alias. */
#define LED0_NODE DT_ALIAS(led0)

/* Stack size for the blinking thread */
#define BLINK_THREAD_STACK_SIZE 512

/* Define stack areas for the threads at compile time */
K_THREAD_STACK_DEFINE(blink_stack, BLINK_THREAD_STACK_SIZE);
K_THREAD_STACK_DEFINE(input_stack, BLINK_THREAD_STACK_SIZE);

/* Declare thread data structs */
static struct k_thread blink_thread;
static struct k_thread input_thread;

/* Define mutex */
K_MUTEX_DEFINE(my_mutex);

/* Define shared blink sleep value */
static int32_t blink_sleep_ms = 500;

/*
 * A build error on this line means your board is unsupported.
 * See the sample documentation for information on how to fix this.
 */
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);


void blink_thread_start(void *arg_1, void *arg_2, void *arg_3)
{
	int ret;
	int32_t sleep_ms;

	while (1)
	{
		/* update sleep time */
		k_mutex_lock(&my_mutex, K_FOREVER);
		sleep_ms = blink_sleep_ms;
		k_mutex_unlock(&my_mutex);
		
		ret = gpio_pin_toggle_dt(&led);
		if (ret < 0) {
			printf("Error: Toggle led failed.\r\n");
		}

		k_msleep(sleep_ms);
	}
}

void input_thread_start(void *arg_1, void *arg_2, void *arg_3)
{
	int32_t inc;

	while (1) {
		/* get line from console (blocking) */
		const char *line = console_getline();

		/* see if first character is + or - */
		if (line[0] == '+')
		{
			inc = 1;
		}
		else if (line[0] == '-')
		{
			inc = -1;
		}
		else
		{
			continue;
		}

		/* update value */
		k_mutex_lock(&my_mutex, K_FOREVER);
		
		blink_sleep_ms += inc * 100;
		if (blink_sleep_ms > BLINK_MAX_SLEEP_TIME_MS)
		{
			blink_sleep_ms = BLINK_MAX_SLEEP_TIME_MS;
		}
		else if (blink_sleep_ms < BLINK_MIN_SLEEP_TIME_MS)
		{
			blink_sleep_ms = BLINK_MIN_SLEEP_TIME_MS;
		}
		k_mutex_unlock(&my_mutex);

	}
}


int main(void)
{
	int ret;
	int32_t sleep_ms;
	k_tid_t blink_tid;
	k_tid_t input_tid;

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

	/* init the console */
	console_getline_init();

		/* Start the blink thread */
	input_tid = k_thread_create(
					&input_thread, 				/* thread struct */
					input_stack,				/* stack */
					K_THREAD_STACK_SIZEOF(input_stack), /* size of the stack */
					input_thread_start,			/* entry point */
					NULL,						/* arg1 */
					NULL,						/* arg2 */
					NULL,						/* arg3 */
					6,							/* priority */
					0,							/* thread options */
					K_NO_WAIT					/* delay */
				);


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
		k_msleep(1000);
		
		/* check current blink delay and print it every seconde */
		k_mutex_lock(&my_mutex, K_FOREVER);
		sleep_ms = blink_sleep_ms;
		k_mutex_unlock(&my_mutex);

		printf("Current blink delay is: %i \r\n", sleep_ms);
	}
	return 0;
}