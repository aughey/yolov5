var mqtt = require('mqtt')
var client = mqtt.connect('mqtt://127.0.0.1')

client.on('error', (err) => {
    console.log(err);
})

client.on('connect', function () {
    console.log("Listening to " + process.argv[2])
    client.subscribe(process.argv[2], function (err) {
        if (!err) {
            client.publish(process.argv[2], 'Hello mqtt')
        }
    })
})


client.on('message', function (topic, message) {
    // message is Buffer
    console.log(topic + ": " + message.toString())
})// JavaScript source code
