// Starfield Animation Script
const canvas = document.getElementById('starfield');
const ctx = canvas.getContext('2d');

// Set canvas size
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Star properties
const stars = [];
const starCount = 200;
const speed = 0.5;

// Initialize stars
for (let i = 0; i < starCount; i++) {
  stars.push({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    radius: Math.random() * 1.5,
    opacity: Math.random() * 0.5 + 0.3,
    speed: Math.random() * speed + 0.1
  });
}

// Animation loop
function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  stars.forEach(star => {
    // Draw star
    ctx.beginPath();
    ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(168, 85, 247, ${star.opacity})`;
    ctx.fill();
    
    // Move star
    star.y += star.speed;
    
    // Reset star when it goes off screen
    if (star.y > canvas.height) {
      star.y = 0;
      star.x = Math.random() * canvas.width;
    }
  });
  
  requestAnimationFrame(animate);
}

animate();