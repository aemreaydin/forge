# Forge Checklist

- [ ] Syncronization2 research
- [ ] Check queue support first and create queues accordingly.
- [ ] Implement Drop for app and make sure everything is destroyed properly
- [ ] Check out allocations and allocation callbacks
- [ ] Check for format and colorspace when creating a swapchain
- [ ] Most places have queue family index hard-coded
- [ ] Create own surface without using any libraries (will need to write platform code)
- [ ] Create proper queues for each queue family (compute, transfer, graphics)
- [ ] Check for Vulkan Instance version when running the application
- [ ] Add better check for physical device selection
- [ ] Look at shader objects and VK_EXT_shader_object
- [ ] Add pipelineCache for graphics pipeline
- [ ] MSAA
- [ ] Graphics pipeline from JSON or YAML
