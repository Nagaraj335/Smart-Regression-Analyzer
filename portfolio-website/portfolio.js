// Portfolio data and configuration
const PORTFOLIO_CONFIG = {
    name: "Nagaraj335",
    title: "Full Stack Developer & Chess Enthusiast",
    github: "https://github.com/Nagaraj335",
    email: "nagaraj935377@gmail.com"
};

// Chess piece mappings to portfolio sections
const PIECE_MAPPING = {
    '♚': { section: 'about', title: 'About Me', tooltip: 'The King - Learn about me' },
    '♔': { section: 'about', title: 'About Me', tooltip: 'The King - Learn about me' },
    '♛': { section: 'skills', title: 'Skills & Expertise', tooltip: 'The Queen - My skills' },
    '♕': { section: 'skills', title: 'Skills & Expertise', tooltip: 'The Queen - My skills' },
    '♜': { section: 'projects', title: 'Major Projects', tooltip: 'The Rook - Major projects' },
    '♖': { section: 'projects', title: 'Major Projects', tooltip: 'The Rook - Major projects' },
    '♝': { section: 'learning', title: 'Learning Journey', tooltip: 'The Bishop - Learning path' },
    '♗': { section: 'learning', title: 'Learning Journey', tooltip: 'The Bishop - Learning path' },
    '♞': { section: 'hobbies', title: 'Fun Projects & Hobbies', tooltip: 'The Knight - Fun projects' },
    '♘': { section: 'hobbies', title: 'Fun Projects & Hobbies', tooltip: 'The Knight - Fun projects' },
    '♟': { section: 'achievements', title: 'Achievements & Certifications', tooltip: 'The Pawn - Achievements' },
    '♙': { section: 'achievements', title: 'Achievements & Certifications', tooltip: 'The Pawn - Achievements' }
};

// Portfolio content sections
const PORTFOLIO_CONTENT = {
    about: {
        icon: '♚',
        content: `
            <div class="space-y-6">
                <div class="flex flex-col md:flex-row items-center gap-6">
                    <div class="w-32 h-32 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center text-6xl">
                        👨‍💻
                    </div>
                    <div class="text-center md:text-left">
                        <h3 class="text-2xl font-bold mb-2">${PORTFOLIO_CONFIG.name}</h3>
                        <p class="text-lg text-gray-300">${PORTFOLIO_CONFIG.title}</p>
                        <p class="text-gray-400 mt-2">Passionate about creating innovative solutions</p>
                        <div class="flex gap-4 mt-4 justify-center md:justify-start">
                            <a href="${PORTFOLIO_CONFIG.github}" target="_blank" class="text-blue-400 hover:text-blue-300">
                                <i class="fab fa-github text-xl"></i>
                            </a>
                            <a href="mailto:${PORTFOLIO_CONFIG.email}" class="text-blue-400 hover:text-blue-300">
                                <i class="fas fa-envelope text-xl"></i>
                            </a>
                        </div>
                    </div>
                </div>
                <div class="grid md:grid-cols-2 gap-6">
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-2 text-yellow-400">🎯 Background</h4>
                        <p class="text-gray-300">Software developer with expertise in web technologies, data science, and AI. I love solving complex problems and building user-friendly applications that make a difference.</p>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-2 text-yellow-400">💡 Interests</h4>
                        <p class="text-gray-300">Chess strategy, Machine Learning, Web Development, Game Development, and creating educational content. Always eager to learn new technologies.</p>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-2 text-yellow-400">🌟 Philosophy</h4>
                        <p class="text-gray-300">"Like chess, programming requires strategic thinking, patience, and the ability to see multiple moves ahead."</p>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-2 text-yellow-400">📍 Location</h4>
                        <p class="text-gray-300">Available for remote work and collaboration on exciting projects worldwide.</p>
                    </div>
                </div>
            </div>
        `
    },
    
    skills: {
        icon: '♛',
        content: `
            <div class="space-y-6">
                <div class="grid md:grid-cols-2 gap-6">
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-4 text-yellow-400">💻 Programming Languages</h4>
                        <div class="space-y-3">
                            ${createSkillBar('Python', 90)}
                            ${createSkillBar('JavaScript', 85)}
                            ${createSkillBar('HTML/CSS', 95)}
                            ${createSkillBar('Java', 70)}
                            ${createSkillBar('SQL', 80)}
                        </div>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-4 text-yellow-400">🛠️ Frameworks & Tools</h4>
                        <div class="space-y-3">
                            ${createSkillBar('React', 80)}
                            ${createSkillBar('Flask/Django', 85)}
                            ${createSkillBar('Node.js', 75)}
                            ${createSkillBar('Git/GitHub', 85)}
                            ${createSkillBar('Docker', 65)}
                        </div>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-4 text-yellow-400">🤖 AI/ML & Data</h4>
                        <div class="space-y-3">
                            ${createSkillBar('Scikit-learn', 80)}
                            ${createSkillBar('Pandas/NumPy', 85)}
                            ${createSkillBar('TensorFlow', 70)}
                            ${createSkillBar('Data Visualization', 85)}
                            ${createSkillBar('Statistical Analysis', 75)}
                        </div>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-4 text-yellow-400">☁️ Cloud & DevOps</h4>
                        <div class="space-y-3">
                            ${createSkillBar('AWS/Azure', 60)}
                            ${createSkillBar('CI/CD', 65)}
                            ${createSkillBar('Linux', 75)}
                            ${createSkillBar('API Development', 85)}
                            ${createSkillBar('Database Design', 80)}
                        </div>
                    </div>
                </div>
            </div>
        `
    },
    
    projects: {
        icon: '♜',
        content: `
            <div class="space-y-6">
                <div class="grid md:grid-cols-2 gap-6">
                    <div class="section-card p-6 rounded-lg">
                        <div class="text-4xl mb-4">♟️</div>
                        <h4 class="text-xl font-semibold mb-2 text-yellow-400">Vexora's Chess Bot</h4>
                        <p class="text-gray-300 mb-4">A sophisticated web-based chess game with intelligent AI opponent featuring ELO-based difficulty system (400-3000 rating). Complete with smooth animations and professional UI.</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <span class="bg-blue-600 px-2 py-1 rounded text-sm">Python</span>
                            <span class="bg-green-600 px-2 py-1 rounded text-sm">Flask</span>
                            <span class="bg-yellow-600 px-2 py-1 rounded text-sm">JavaScript</span>
                            <span class="bg-purple-600 px-2 py-1 rounded text-sm">React</span>
                        </div>
                        <div class="flex gap-4">
                            <a href="${PORTFOLIO_CONFIG.github}/Chess-Bot" target="_blank" class="text-blue-400 hover:text-blue-300">
                                <i class="fab fa-github mr-1"></i>View Code
                            </a>
                            <a href="#" class="text-green-400 hover:text-green-300">
                                <i class="fas fa-external-link-alt mr-1"></i>Live Demo
                            </a>
                        </div>
                    </div>
                    
                    <div class="section-card p-6 rounded-lg">
                        <div class="text-4xl mb-4">📊</div>
                        <h4 class="text-xl font-semibold mb-2 text-yellow-400">Data Science Projects</h4>
                        <p class="text-gray-300 mb-4">Various machine learning and data analysis projects including predictive models, data visualization, and statistical analysis.</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <span class="bg-blue-600 px-2 py-1 rounded text-sm">Python</span>
                            <span class="bg-orange-600 px-2 py-1 rounded text-sm">Pandas</span>
                            <span class="bg-red-600 px-2 py-1 rounded text-sm">Scikit-learn</span>
                            <span class="bg-green-600 px-2 py-1 rounded text-sm">Jupyter</span>
                        </div>
                        <a href="${PORTFOLIO_CONFIG.github}?tab=repositories" target="_blank" class="text-blue-400 hover:text-blue-300">
                            <i class="fab fa-github mr-1"></i>View Projects
                        </a>
                    </div>
                    
                    <div class="section-card p-6 rounded-lg">
                        <div class="text-4xl mb-4">🌐</div>
                        <h4 class="text-xl font-semibold mb-2 text-yellow-400">Web Applications</h4>
                        <p class="text-gray-300 mb-4">Full-stack web applications with modern frameworks, responsive design, and interactive user experiences.</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <span class="bg-cyan-600 px-2 py-1 rounded text-sm">React</span>
                            <span class="bg-green-600 px-2 py-1 rounded text-sm">Node.js</span>
                            <span class="bg-blue-600 px-2 py-1 rounded text-sm">TailwindCSS</span>
                            <span class="bg-gray-600 px-2 py-1 rounded text-sm">MongoDB</span>
                        </div>
                        <a href="#" class="text-blue-400 hover:text-blue-300">
                            View Portfolio →
                        </a>
                    </div>
                    
                    <div class="section-card p-6 rounded-lg">
                        <div class="text-4xl mb-4">🤖</div>
                        <h4 class="text-xl font-semibold mb-2 text-yellow-400">AI/ML Projects</h4>
                        <p class="text-gray-300 mb-4">Machine learning models, chatbots, and AI-powered applications solving real-world problems.</p>
                        <div class="flex flex-wrap gap-2 mb-4">
                            <span class="bg-orange-600 px-2 py-1 rounded text-sm">TensorFlow</span>
                            <span class="bg-blue-600 px-2 py-1 rounded text-sm">PyTorch</span>
                            <span class="bg-green-600 px-2 py-1 rounded text-sm">OpenAI API</span>
                            <span class="bg-purple-600 px-2 py-1 rounded text-sm">Hugging Face</span>
                        </div>
                        <a href="#" class="text-blue-400 hover:text-blue-300">
                            Explore AI Projects →
                        </a>
                    </div>
                </div>
            </div>
        `
    },
    
    learning: {
        icon: '♝',
        content: `
            <div class="space-y-6">
                <div class="grid md:grid-cols-3 gap-4">
                    <div class="section-card p-4 rounded-lg text-center">
                        <div class="text-3xl mb-2">🎓</div>
                        <h4 class="font-semibold text-yellow-400">Education</h4>
                        <p class="text-gray-300 text-sm mt-2">Computer Science & Engineering</p>
                        <p class="text-gray-400 text-xs mt-1">Continuous Learning Mindset</p>
                    </div>
                    <div class="section-card p-4 rounded-lg text-center">
                        <div class="text-3xl mb-2">📚</div>
                        <h4 class="font-semibold text-yellow-400">Current Focus</h4>
                        <p class="text-gray-300 text-sm mt-2">Advanced AI & Cloud Architecture</p>
                        <p class="text-gray-400 text-xs mt-1">Microservices & Kubernetes</p>
                    </div>
                    <div class="section-card p-4 rounded-lg text-center">
                        <div class="text-3xl mb-2">🎯</div>
                        <h4 class="font-semibold text-yellow-400">Next Goals</h4>
                        <p class="text-gray-300 text-sm mt-2">Full-Stack AI Developer</p>
                        <p class="text-gray-400 text-xs mt-1">Technical Leadership</p>
                    </div>
                </div>
                
                <div class="section-card p-6 rounded-lg">
                    <h4 class="text-xl font-semibold mb-4 text-yellow-400">🛤️ Learning Timeline</h4>
                    <div class="space-y-4">
                        <div class="border-l-4 border-blue-500 pl-4">
                            <h5 class="font-semibold">Foundation Phase</h5>
                            <p class="text-gray-300 text-sm">HTML, CSS, JavaScript fundamentals</p>
                            <p class="text-gray-400 text-xs">Programming logic and problem-solving</p>
                        </div>
                        <div class="border-l-4 border-green-500 pl-4">
                            <h5 class="font-semibold">Web Development</h5>
                            <p class="text-gray-300 text-sm">React, Node.js, APIs, Databases</p>
                            <p class="text-gray-400 text-xs">Full-stack application development</p>
                        </div>
                        <div class="border-l-4 border-purple-500 pl-4">
                            <h5 class="font-semibold">Data Science & AI</h5>
                            <p class="text-gray-300 text-sm">Python, ML algorithms, Deep Learning</p>
                            <p class="text-gray-400 text-xs">Statistical analysis and model deployment</p>
                        </div>
                        <div class="border-l-4 border-yellow-500 pl-4">
                            <h5 class="font-semibold">Advanced Topics</h5>
                            <p class="text-gray-300 text-sm">Cloud computing, DevOps, System design</p>
                            <p class="text-gray-400 text-xs">Scalable architecture and best practices</p>
                        </div>
                    </div>
                </div>
                
                <div class="grid md:grid-cols-2 gap-6">
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-3 text-yellow-400">🔥 Currently Learning</h4>
                        <ul class="space-y-2 text-gray-300">
                            <li>• Advanced React patterns and hooks</li>
                            <li>• Microservices architecture</li>
                            <li>• Cloud-native development</li>
                            <li>• Advanced ML algorithms</li>
                        </ul>
                    </div>
                    <div class="section-card p-4 rounded-lg">
                        <h4 class="font-semibold mb-3 text-yellow-400">📅 Learning Resources</h4>
                        <ul class="space-y-2 text-gray-300">
                            <li>• Online courses and tutorials</li>
                            <li>• Open source contributions</li>
                            <li>• Technical books and papers</li>
                            <li>• Community forums and blogs</li>
                        </ul>
                    </div>
                </div>
            </div>
        `
    },
    
    hobbies: {
        icon: '♞',
        content: `
            <div class="space-y-6">
                <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">♟️</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">Chess</h4>
                        <p class="text-gray-300 text-sm">Strategic thinking and pattern recognition. Love analyzing games and improving tactics.</p>
                    </div>
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">🎮</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">Game Development</h4>
                        <p class="text-gray-300 text-sm">Creating interactive experiences and experimenting with game mechanics.</p>
                    </div>
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">🤖</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">AI Experiments</h4>
                        <p class="text-gray-300 text-sm">Building intelligent systems and exploring machine learning applications.</p>
                    </div>
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">📱</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">Mobile Apps</h4>
                        <p class="text-gray-300 text-sm">Cross-platform development and mobile user experience design.</p>
                    </div>
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">🌐</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">Web Experiments</h4>
                        <p class="text-gray-300 text-sm">Interactive web applications and creative coding projects.</p>
                    </div>
                    <div class="section-card p-6 rounded-lg text-center hover:scale-105 transition-transform">
                        <div class="text-4xl mb-4">📊</div>
                        <h4 class="text-lg font-semibold mb-2 text-yellow-400">Data Visualization</h4>
                        <p class="text-gray-300 text-sm">Beautiful data storytelling and interactive dashboards.</p>
                    </div>
                </div>
                
                <div class="section-card p-6 rounded-lg">
                    <h4 class="text-xl font-semibold mb-4 text-yellow-400">🎪 Fun Side Projects</h4>
                    <div class="grid md:grid-cols-2 gap-4">
                        <div class="bg-gray-700 p-4 rounded">
                            <h5 class="font-semibold mb-2">🎲 Random Quote Generator</h5>
                            <p class="text-gray-300 text-sm">Motivational quotes with beautiful animations</p>
                        </div>
                        <div class="bg-gray-700 p-4 rounded">
                            <h5 class="font-semibold mb-2">🎵 Music Player</h5>
                            <p class="text-gray-300 text-sm">Custom web-based music player with visualizations</p>
                        </div>
                        <div class="bg-gray-700 p-4 rounded">
                            <h5 class="font-semibold mb-2">🌤️ Weather App</h5>
                            <p class="text-gray-300 text-sm">Beautiful weather dashboard with forecasting</p>
                        </div>
                        <div class="bg-gray-700 p-4 rounded">
                            <h5 class="font-semibold mb-2">🎨 Color Palette Generator</h5>
                            <p class="text-gray-300 text-sm">AI-powered color scheme generator for designers</p>
                        </div>
                    </div>
                </div>
            </div>
        `
    },
    
    achievements: {
        icon: '♟',
        content: `
            <div class="space-y-6">
                <div class="grid md:grid-cols-2 gap-6">
                    <div class="section-card p-6 rounded-lg">
                        <h4 class="text-xl font-semibold mb-4 text-yellow-400">🏆 Key Achievements</h4>
                        <div class="space-y-3">
                            <div class="flex items-start gap-3">
                                <div class="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                                <div>
                                    <h5 class="font-semibold">Chess AI Engine</h5>
                                    <p class="text-gray-300 text-sm">Successfully built and deployed a sophisticated chess AI with ELO rating system</p>
                                </div>
                            </div>
                            <div class="flex items-start gap-3">
                                <div class="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                                <div>
                                    <h5 class="font-semibold">Full-Stack Applications</h5>
                                    <p class="text-gray-300 text-sm">Created multiple end-to-end web applications with modern tech stacks</p>
                                </div>
                            </div>
                            <div class="flex items-start gap-3">
                                <div class="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                                <div>
                                    <h5 class="font-semibold">ML Model Deployment</h5>
                                    <p class="text-gray-300 text-sm">Implemented and deployed machine learning models in production</p>
                                </div>
                            </div>
                            <div class="flex items-start gap-3">
                                <div class="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                                <div>
                                    <h5 class="font-semibold">Open Source Contributions</h5>
                                    <p class="text-gray-300 text-sm">Active contributor to open-source projects and communities</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section-card p-6 rounded-lg">
                        <h4 class="text-xl font-semibold mb-4 text-yellow-400">📜 Certifications & Learning</h4>
                        <div class="space-y-3">
                            <div class="border-l-4 border-blue-500 pl-3">
                                <h5 class="font-semibold">Full-Stack Web Development</h5>
                                <p class="text-gray-400 text-sm">Modern JavaScript, React, Node.js</p>
                            </div>
                            <div class="border-l-4 border-green-500 pl-3">
                                <h5 class="font-semibold">Python Programming</h5>
                                <p class="text-gray-400 text-sm">Advanced Python and Data Structures</p>
                            </div>
                            <div class="border-l-4 border-purple-500 pl-3">
                                <h5 class="font-semibold">Data Science & ML</h5>
                                <p class="text-gray-400 text-sm">Machine Learning and Data Analytics</p>
                            </div>
                            <div class="border-l-4 border-yellow-500 pl-3">
                                <h5 class="font-semibold">Cloud Computing</h5>
                                <p class="text-gray-400 text-sm">AWS/Azure Cloud Services</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section-card p-6 rounded-lg">
                    <h4 class="text-xl font-semibold mb-4 text-yellow-400">📈 Skills Progress</h4>
                    <div class="grid md:grid-cols-3 gap-6">
                        <div class="text-center">
                            <div class="text-3xl font-bold text-blue-400">15+</div>
                            <p class="text-gray-300">Projects Completed</p>
                        </div>
                        <div class="text-center">
                            <div class="text-3xl font-bold text-green-400">5+</div>
                            <p class="text-gray-300">Technologies Mastered</p>
                        </div>
                        <div class="text-center">
                            <div class="text-3xl font-bold text-purple-400">100+</div>
                            <p class="text-gray-300">GitHub Contributions</p>
                        </div>
                    </div>
                </div>
                
                <div class="section-card p-6 rounded-lg">
                    <h4 class="text-xl font-semibold mb-4 text-yellow-400">🎯 Future Goals</h4>
                    <div class="grid md:grid-cols-2 gap-4">
                        <div>
                            <h5 class="font-semibold mb-2">Technical Goals</h5>
                            <ul class="space-y-1 text-gray-300 text-sm">
                                <li>• Master advanced AI/ML techniques</li>
                                <li>• Contribute to major open-source projects</li>
                                <li>• Build scalable distributed systems</li>
                            </ul>
                        </div>
                        <div>
                            <h5 class="font-semibold mb-2">Career Goals</h5>
                            <ul class="space-y-1 text-gray-300 text-sm">
                                <li>• Lead innovative technology projects</li>
                                <li>• Mentor upcoming developers</li>
                                <li>• Create impactful solutions</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `
    }
};

// Utility function to create skill bars
function createSkillBar(skill, percentage) {
    return `
        <div class="skill-item">
            <div class="flex justify-between mb-1">
                <span>${skill}</span>
                <span>${percentage}%</span>
            </div>
            <div class="w-full bg-gray-600 rounded-full h-2">
                <div class="skill-bar h-2 rounded-full transition-all duration-1000 ease-out" style="width: ${percentage}%"></div>
            </div>
        </div>
    `;
}

// Animation utilities
const AnimationUtils = {
    // Animate skill bars when they come into view
    animateSkillBars() {
        const skillBars = document.querySelectorAll('.skill-bar');
        skillBars.forEach((bar, index) => {
            setTimeout(() => {
                bar.style.width = bar.style.width || '0%';
                const targetWidth = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 100);
            }, index * 100);
        });
    },

    // Animate cards entrance
    animateCards() {
        const cards = document.querySelectorAll('.section-card');
        gsap.fromTo(cards, 
            { opacity: 0, y: 30 },
            { 
                opacity: 1, 
                y: 0, 
                duration: 0.6, 
                stagger: 0.1,
                ease: "power2.out"
            }
        );
    },

    // Piece click animation
    animatePieceClick(element) {
        gsap.to(element, {
            scale: 1.2,
            rotation: 360,
            duration: 0.6,
            ease: "back.out(1.7)",
            yoyo: true,
            repeat: 1
        });
    }
};

// Export for use in main HTML file
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PORTFOLIO_CONFIG,
        PIECE_MAPPING,
        PORTFOLIO_CONTENT,
        createSkillBar,
        AnimationUtils
    };
}
